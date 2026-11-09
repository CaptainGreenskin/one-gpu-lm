"""Execute one frozen base/train/candidate quality screen on an already prepared Linux root."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

# Also supports invocation by file path from a prepared repository root.
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.checkpoints import load_checkpoint
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, gpu_lock_path, single_writer, utc
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.serving.client import request


def terminate(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();out=args.out.resolve();os.chdir(root)
    if out.exists():raise ValueError('Preserve previous screen; inspect evidence before recovery')
    exp=root/'experiments/014-coding-quality'
    protocol=json.loads((exp/'protocol.json').read_text())
    if protocol['status']!='frozen_before_model_rollouts':raise ValueError('Freeze protocol first')
    frozen=protocol['frozen_files']
    for name,digest in frozen.items():
        if sha256(root/name)!=digest:raise ValueError('Frozen screen input changed: '+name)
    runtime=json.loads((root/'runtime.json').read_text())
    inputs=json.loads((root/'private/inputs.json').read_text())
    if sha256(Path(inputs['ONE_GPU_TRAIN_DATA']))!=protocol['prepared_data_sha256']:
        raise ValueError('Training partition differs from frozen prepared data')
    env=dict(os.environ,**inputs,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    gpu=runtime['model_python'];cpu=str(root/'.venv-agent/bin/python')
    out.mkdir(parents=True);(out/'private').mkdir(mode=0o700)
    state={'schema_version':1,'started_utc':utc(),'started_epoch':time.time(),'complete':False,
        'protocol_sha256':sha256(exp/'protocol.json'),'stages':[]}
    limits=protocol['execution'];deadline=time.monotonic()+limits['max_pipeline_seconds']
    def stage(name,**fields):
        state.update(stage=name,updated_utc=utc(),**fields);atomic_json(out/'controller.json',state)
        print(name,flush=True)
    def remaining():
        seconds=deadline-time.monotonic()
        if seconds<=0:raise TimeoutError('Frozen pipeline time budget exhausted')
        return seconds
    def run(name,command,timeout):
        stage(name);start=time.monotonic();process=None
        try:
            with (out/'private'/f'{name}.log').open('w') as log:
                process=subprocess.Popen(command,stdout=log,stderr=log,env=env,start_new_session=True)
                code=process.wait(timeout=min(timeout,remaining()))
                if code:raise RuntimeError('Stage failed: '+name)
        finally:terminate(process)
        state['stages'].append({'name':name,'seconds':time.monotonic()-start});stage(name+'_finished')
    def evaluate(variant,adapter=None,adapter_hashes=None):
        backend=None
        with single_writer(gpu_lock_path()),tempfile.TemporaryDirectory(prefix='oglm014-') as temp:
            sock=Path(temp)/'model.sock';backend_out=out/'private'/('backend-'+variant)
            command=[gpu,'-m','one_gpu_lm','serve','--model',runtime['model'],'--profile',str(exp/'model-profile.json'),
                '--socket',str(sock),'--out',str(backend_out)]
            if adapter:command+=['--adapter',str(adapter)]
            stage('loading_'+variant)
            with (out/'private'/('backend-'+variant+'.log')).open('w') as log:
                try:
                    backend=subprocess.Popen(command,env=env,stdout=log,stderr=log,start_new_session=True)
                    ready_deadline=time.monotonic()+min(limits['max_backend_start_seconds'],remaining())
                    while not (backend_out/'ready.json').exists():
                        if backend.poll() is not None:raise RuntimeError('Backend failed before readiness')
                        if time.monotonic()>ready_deadline:raise TimeoutError('Backend readiness timeout')
                        time.sleep(1)
                    health=request(sock,{'operation':'health'})
                    if health['profile_sha256']!=sha256(exp/'model-profile.json') or health['adapter_sha256']!=(adapter_hashes or {}):
                        raise ValueError('Serving artifact identity mismatch')
                    atomic_json(out/('artifact-'+variant+'.json'),health)
                    if variant=='baseline':
                        run('loaded-controls',[cpu,'-m','one_gpu_lm.evaluation.preflight','--root',str(root),
                            '--tasks',str(root/'private/tasks.json'),'--out',str(out/'loaded-controls'),'--clean-only'],300)
                        control=json.loads((out/'loaded-controls/admission.json').read_text())
                        if len(control['clean'])!=protocol['repository_count'] or not all(r['passed'] for r in control['clean']):
                            raise ValueError('GPU-loaded clean controls failed; frozen scoring cannot start')
                    run(variant,[cpu,'-m','one_gpu_lm.evaluation.campaign','--root',str(root),'--tasks',str(root/'private/tasks.json'),
                        '--admission',str(root/'runs/admission/admission.json'),'--protocol',str(exp/'protocol.json'),
                        '--socket',str(sock),'--out',str(out/variant),'--split','dev','test'],limits['max_campaign_seconds'])
                    run('legacy-'+variant,[cpu,'-m','one_gpu_lm.evaluation.regression','--tasks',str(root/'private/legacy-functions.jsonl'),
                        '--socket',str(sock),'--node',str(Path(runtime['node_root'])/'bin/node'),'--out',str(out/('legacy-'+variant))],limits['max_legacy_seconds'])
                finally:terminate(backend)
    def interrupted(signum,frame):raise KeyboardInterrupt('Controller interrupted')
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupted)
    monitor=Telemetry(out/'telemetry.csv',1,5)
    with single_writer(root/'private/quality.lock'):
        monitor.start()
        try:
            run('doctor',[gpu,'-m','one_gpu_lm','doctor','--config',str(exp/'train-config.json'),'--out',str(out/'doctor.json')],300)
            evaluate('baseline')
            run('training',[gpu,'-m','one_gpu_lm','train','--config',str(exp/'train-config.json'),'--out',str(out/'candidate')],limits['max_training_seconds'])
            trained=json.loads((out/'candidate/result.json').read_text())
            if trained['status']!='completed' or trained['steps']!=protocol['execution']['max_optimizer_updates'] or not trained['saved_reload_greedy_ids_equal']:
                raise ValueError('Candidate did not meet fixed training contract')
            identity=json.loads((out/'candidate/identity.json').read_text())
            checkpoint,_,manifest=load_checkpoint(out/'candidate',identity)
            hashes={k.removeprefix('adapter/'):v for k,v in manifest['files'].items() if k.startswith('adapter/') and k.endswith('.safetensors')}
            evaluate('adapted',checkpoint/'adapter',hashes)
            stage('completed',complete=True,finished_utc=utc(),finished_epoch=time.time())
        except BaseException as error:
            stage('failed',error_type=type(error).__name__,finished_utc=utc(),finished_epoch=time.time());raise
        finally:
            state['telemetry']=monitor.stop();atomic_json(out/'controller.json',state)


if __name__=='__main__':main()
