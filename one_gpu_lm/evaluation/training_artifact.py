"""Run legacy function regression through an owned local backend for a verified training artifact."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.runstate import gpu_lock_path, single_writer
from one_gpu_lm.serving.client import request
from one_gpu_lm.common.checkpoints import load_checkpoint
from one_gpu_lm.models.verify import verify_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--node',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variant',choices=['base','adapted'],default='adapted')
    args=p.parse_args();run=args.run.resolve();out=args.out.resolve()
    if out.exists():raise ValueError('Preserve existing regression evidence')
    identity=json.loads((run/'identity.json').read_text())
    if json.loads((run/'result.json').read_text())['status']!='completed':raise ValueError('Training has not completed')
    checkpoint,_,manifest=load_checkpoint(run,identity)
    locations=json.loads((run/'private/input-locations.json').read_text())
    if sha256(run/'profile.json')!=identity['input_hashes']['profile']:raise ValueError('Profile changed')
    if sha256(Path(locations['weights_manifest']))!=identity['input_hashes']['weights_manifest']:
        raise ValueError('Base weights manifest changed')
    out.mkdir(parents=True);(out/'private').mkdir(mode=0o700)
    with single_writer(gpu_lock_path()),tempfile.TemporaryDirectory(prefix='oglm-backend-') as temp:
        verified=verify_model(Path(locations['model_dir']),Path(locations['weights_manifest']),identity['profile'])
        socket=Path(temp)/'model.sock'
        command=[sys.executable,'-m','one_gpu_lm.serving.local_backend','--model',locations['model_dir'],
                 '--profile',str(run/'profile.json'),'--socket',str(socket),'--out',str(out/'private/backend')]
        if args.variant=='adapted':command+=['--adapter',str(checkpoint/'adapter')]
        with (out/'private/backend.log').open('w') as log:
            backend=subprocess.Popen(command,stdout=log,stderr=log)
            try:
                for _ in range(180):
                    if backend.poll() is not None:raise RuntimeError('Owned model backend failed to load')
                    if (out/'private/backend/ready.json').exists():break
                    time.sleep(1)
                else:raise TimeoutError('Owned backend readiness deadline exceeded')
                health=request(socket,{'operation':'health'})
                expected={name.removeprefix('adapter/'):digest for name,digest in manifest['files'].items()
                          if name.startswith('adapter/') and name.endswith('.safetensors')} if args.variant=='adapted' else {}
                if health['adapter_sha256']!=expected or health['profile_sha256']!=sha256(run/'profile.json'):
                    raise ValueError('Backend loaded a different artifact')
                subprocess.run([sys.executable,'-m','one_gpu_lm.evaluation.regression','--tasks',str(args.tasks.resolve()),
                    '--socket',str(socket),'--node',str(args.node.resolve()),'--out',str(out/'metrics')],check=True,timeout=1800)
                write_json(out/'artifact.json',{'variant':args.variant,'training_identity_sha256':json.loads((run/'result.json').read_text())['identity_sha256'],
                    'base_content_verification':verified,'backend':health,'task_sha256':sha256(args.tasks),'purpose':'function-regression-smoke-not-daily-model-acceptance'})
            finally:
                if backend.poll() is None:
                    backend.terminate()
                    try:backend.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        backend.kill();backend.wait(timeout=10)


if __name__=='__main__':main()
