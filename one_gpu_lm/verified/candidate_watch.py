"""Enforce the one-candidate wall-clock cap; a killed partial run is ineligible."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from one_gpu_lm.common.receipts import sha256
from .rollout import atomic
from .tasks import relative


def run(root, protocol_path):
    root,protocol_path=Path(root).resolve(),Path(protocol_path).resolve()
    protocol=json.loads(protocol_path.read_text())
    spec=json.loads((root/relative(protocol['cycle_spec'])).read_text())
    receipt=root/relative(protocol['supervisor_receipt']);out=root/relative(protocol['run_directory'])
    if receipt.exists() or out.exists():
        raise ValueError('Candidate attempt already exists')
    state={'started_epoch':time.time(),'protocol_sha256':sha256(protocol_path),'complete':False}
    with receipt.open('x') as stream:stream.write(json.dumps(state)+'\n')
    runtime=json.loads((root/'private/parent-runtime.json').read_text())
    child=subprocess.Popen([runtime['model_python'],'-u','-m','one_gpu_lm.verified.candidate',
              '--root',str(root),'--protocol',str(protocol_path)],cwd=root,
              stdout=(root/'private/candidate-worker.log').open('wb'),stderr=subprocess.STDOUT,start_new_session=True)
    state['worker_pid']=child.pid;atomic(receipt,state);timed_out=False
    try:
        child.wait(timeout=spec['candidate']['maximum_training_pipeline_seconds'])
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
    state.update(complete=True,timed_out=timed_out,worker_returncode=child.returncode,
                 finished_epoch=time.time(),finished_utc=datetime.now(timezone.utc).isoformat())
    path=out/'result.json';result=json.loads(path.read_text()) if path.exists() else {}
    if timed_out or child.returncode!=0:
        result.update(complete=False,stage='supervisor_timeout' if timed_out else 'worker_failed',
                      finished_epoch=state['finished_epoch'],finished_utc=state['finished_utc'],
                      telemetry_may_be_partial=True,weights_public=False)
        out.mkdir(parents=True,exist_ok=True);atomic(path,result)
    state['candidate_eligible']=result.get('complete',False) and not timed_out and child.returncode==0
    atomic(receipt,state)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.root,a.protocol)))
