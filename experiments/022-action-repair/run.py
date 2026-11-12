"""Train one frozen early-action curriculum and gate inherited sealed tests.

This controller never uploads weights and never declares a release accepted.
Final execution/rights/export audits run after scoring, before any publication.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.checkpoints import load_checkpoint
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, gpu_lock_path, single_writer, utc
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.serving.client import request
from decision import paired, select


def terminate(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)


def require_public_preregistration(root,exp):
    receipt=json.loads((root/'private/public-preregistration.json').read_text())
    commit=receipt.get('commit','')
    if (len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit)
            or receipt.get('anonymous_files_verified') is not True
            or receipt.get('ci_conclusion')!='success'
            or receipt.get('protocol_sha256')!=sha256(exp/'protocol.json')):
        raise ValueError('Require public byte verification and passing CI for this exact frozen protocol')


def execute(root,out,resume=False):
    # Lock before reading/updating attempt counters, not just before model work.
    with single_writer(Path(root).resolve()/'private/controller-state022.lock'):
        return _execute_locked(root,out,resume)


def _execute_locked(root,out,resume=False):
    root=Path(root).resolve();out=Path(out).resolve();os.chdir(root)
    exp=root/'experiments/022-action-repair'
    protocol=json.loads((exp/'protocol.json').read_text())
    plan=json.loads((exp/'preparation-plan.json').read_text())
    if protocol['status']!='frozen_before_model_rollouts':raise ValueError('Freeze protocol first')
    for name,digest in protocol['frozen_files'].items():
        if sha256(root/name)!=digest:raise ValueError('Frozen artifact changed: '+name)
    require_public_preregistration(root,exp)
    runtime=json.loads((root/'runtime.json').read_text());inputs=json.loads((root/'private/inputs.json').read_text())
    if out.exists()!=resume:raise ValueError('Existing runs require explicit resume; new runs must not use it')
    if resume:
        state=json.loads((out/'controller.json').read_text())
        if state['complete']:raise ValueError('Completed campaigns are immutable')
        if state['protocol_sha256']!=sha256(exp/'protocol.json'):raise ValueError('Cannot resume changed protocol')
        state['attempt']+=1
    else:
        out.mkdir(parents=True,mode=0o700);(out/'private').mkdir(mode=0o700)
        now=time.time();state={'schema_version':1,'started_utc':utc(),'started_epoch':now,
            'deadline_epoch':now+plan['budget']['max_gpu_pipeline_wall_seconds'],
            'protocol_sha256':sha256(exp/'protocol.json'),'complete':False,'attempt':1,'stages':[]}
        atomic_json(out/'controller.json',state)
    deadline=time.monotonic()+max(0,state['deadline_epoch']-time.time())
    env=dict(os.environ,**inputs,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    gpu=runtime['model_python'];cpu=str(root/'.venv-agent/bin/python')
    tasks=json.loads((root/'private/tasks.json').read_text())['tasks']
    development=[t['id'] for t in tasks if t['split']=='dev']
    final=[t['id'] for t in tasks if t['split']=='test']
    historical=[t['id'] for t in tasks if t.get('evaluation_role')=='historical_final_regression_not_selection']
    fresh=sorted(set(final)-set(historical))
    if (len(development),len(historical),len(fresh))!=(15,48,24):raise ValueError('Frozen cohort counts differ')
    def remaining():
        seconds=deadline-time.monotonic()
        if seconds<=0:raise TimeoutError('Original 12-hour wall deadline exhausted')
        return seconds
    def stage(name,**extra):
        state.update(stage=name,updated_utc=utc(),**extra);atomic_json(out/'controller.json',state)
        print(name,flush=True)
    def run(name,command,timeout):
        stage(name);start=time.monotonic();process=None
        try:
            with (out/'private'/f'{name}-attempt-{state["attempt"]}.log').open('w') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=stream,env=env,start_new_session=True)
                code=process.wait(timeout=min(timeout,remaining()))
                if code:raise RuntimeError('Stage failed: '+name)
        finally:terminate(process)
        state['stages'].append({'name':name,'seconds':time.monotonic()-start,'attempt':state['attempt']});stage(name+'_finished')
    def adapter(name):
        if name=='reference020':
            path=Path(runtime['reference_adapter'])
            digest=sha256(path/'adapter_model.safetensors')
            if digest!=protocol['reference_adapter']['adapter_model_sha256']:
                raise ValueError('020 comparator weight identity changed')
            return path,{'adapter_model.safetensors':digest}
        path=out/('train-'+name);result=json.loads((path/'result.json').read_text())
        if result['status']!='completed' or result['steps']!=protocol['training_records'] or not result['saved_reload_greedy_ids_equal']:
            raise ValueError('Training artifact did not satisfy frozen recipe')
        checkpoint,_,manifest=load_checkpoint(path,json.loads((path/'identity.json').read_text()))
        hashes={k.removeprefix('adapter/'):v for k,v in manifest['files'].items() if k.startswith('adapter/') and k.endswith('.safetensors')}
        return checkpoint/'adapter',hashes
    def evaluate(label,split,candidate=None,legacy=False):
        campaign_path=out/label/'campaign.json'
        already=campaign_path.exists() and json.loads(campaign_path.read_text())['complete']
        legacy_path=out/('legacy-'+label)/'result.json'
        legacy_done=legacy_path.exists() and json.loads(legacy_path.read_text()).get('complete') is True
        if already and (not legacy or legacy_done):return json.loads(campaign_path.read_text())
        model_adapter,hashes=adapter(candidate) if candidate else (None,{})
        backend=None;attempt=state['attempt']
        with single_writer(gpu_lock_path()),tempfile.TemporaryDirectory(prefix='oglm022-') as temp:
            sock=Path(temp)/'model.sock';backend_out=out/'private'/f'backend-{label}-attempt-{attempt}'
            if backend_out.exists():raise ValueError('Backend attempt directory already exists')
            command=[gpu,'-m','one_gpu_lm','serve','--model',runtime['model'],'--profile',str(exp/'profile-action.json'),
                '--socket',str(sock),'--out',str(backend_out)]
            if model_adapter:command+=['--adapter',str(model_adapter)]
            stage('loading_'+label)
            with (out/'private'/f'backend-{label}-attempt-{attempt}.log').open('w') as stream:
                try:
                    backend=subprocess.Popen(command,env=env,stdout=stream,stderr=stream,start_new_session=True)
                    ready=time.monotonic()+min(plan['budget']['max_backend_start_seconds'],remaining())
                    while not (backend_out/'ready.json').exists():
                        if backend.poll() is not None:raise RuntimeError('Backend failed before readiness')
                        if time.monotonic()>ready:raise TimeoutError('Backend readiness timed out')
                        time.sleep(1)
                    health=request(sock,{'operation':'health'})
                    if health['profile_sha256']!=sha256(exp/'profile-action.json') or health['adapter_sha256']!=hashes:
                        raise ValueError('Serving artifact identity mismatch')
                    atomic_json(out/f'artifact-{label}-attempt-{attempt}.json',health)
                    if label=='loaded-base' and not (out/'loaded-controls/admission.json').exists():
                        run('loaded-controls',[cpu,'-m','one_gpu_lm.evaluation.preflight','--root',str(root),
                            '--tasks',str(root/'private/tasks.json'),'--out',str(out/'loaded-controls'),'--clean-only'],1200)
                    controls=json.loads((out/'loaded-controls/admission.json').read_text())
                    if not controls['complete'] or len(controls['clean'])!=protocol['repository_count'] or not all(r['passed'] for r in controls['clean']):
                        raise ValueError('GPU-loaded clean controls are incomplete or failed')
                    if label=='loaded-base':
                        return controls
                    if not already:
                        command=[cpu,'-m','one_gpu_lm.evaluation.campaign','--root',str(root),
                            '--tasks',str(root/'private/tasks.json'),'--admission',str(root/'runs/admission/admission.json'),
                            '--protocol',str(exp/'protocol.json'),'--socket',str(sock),'--out',str(out/label),
                            '--split',split,'--variant','shared']
                        if campaign_path.exists():command+=['--resume']
                        run(label,command,10800 if split=='test' else 4800)
                    if legacy and not legacy_done:
                        run('legacy-'+label,[cpu,'-m','one_gpu_lm.evaluation.regression',
                            '--tasks',str(root/'private/legacy-functions.jsonl'),'--socket',str(sock),
                            '--node',str(Path(runtime['node_root'])/'bin/node'),'--out',str(out/('legacy-'+label))],1200)
                finally:terminate(backend)
        return json.loads(campaign_path.read_text())
    def interrupted(signum,frame):raise KeyboardInterrupt('Controller interrupted')
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupted)
    monitor=Telemetry(out/f'telemetry-attempt-{state["attempt"]}.csv',state['attempt'],5)
    with single_writer(root/'private/release022.lock'):
        remaining();monitor.start()
        try:
            run('preparation-verify',[gpu,str(exp/'preparation.py'),'--root',str(root),
                '--out',str(out/'private'/f'preparation-attempt-{state["attempt"]}.json')],600)
            derived=root/'runs/reference-data022/derived-data.json'
            evidence=json.loads(derived.read_text())
            if evidence['coverage_passed'] is not True:raise ValueError('CPU-derived coverage must pass before the model pipeline')
            if not (out/'loaded-controls/admission.json').exists():evaluate('loaded-base','train')
            prepared=root/'runs/prepared-data022/action'
            for name,key in [('train.jsonl','prepared_data_sha256'),('receipt.json','prepared_receipt_sha256'),('selection.json','selection_sha256')]:
                if sha256(prepared/name)!=evidence[key]:raise ValueError('Exact frozen training inputs changed')
            # No evaluation generations occur before the single candidate is fixed.
            for name in ['action']:
                path=out/('train-'+name)
                if not ((path/'result.json').exists() and json.loads((path/'result.json').read_text())['status']=='completed'):
                    command=[gpu,'-m','one_gpu_lm','train','--config',str(exp/('train-'+name+'.json')),'--out',str(path)]
                    if path.exists():command+=['--resume']
                    run('train-'+name,command,plan['budget']['max_training_seconds_per_candidate'])
                adapter(name)
            baseline=evaluate('dev-base','dev')
            reference=evaluate('dev-reference020','dev','reference020')
            action=evaluate('dev-action','dev','action')
            selection=select(baseline,reference,action,development,plan);atomic_json(out/'selection.json',selection)
            if selection['selected'] is None:
                atomic_json(out/'release-decision.json',{'accepted':False,'reason':'development_gate_failed','final_test_opened':False,'weights_public':False})
            else:
                selection['final_test_opened']=True;atomic_json(out/'selection.json',selection)
                base_final=evaluate('final-base','test',legacy=True)
                candidate_final=evaluate('final-selected','test',selection['selected'],legacy=True)
                def subset(campaign,ids):return dict(campaign,results=[r for r in campaign['results'] if r['id'] in ids])
                comparisons={name:paired(subset(base_final,ids),subset(candidate_final,ids),ids)
                    for name,ids in [('historical',historical),('fresh',fresh)]}
                atomic_json(out/'comparisons.json',comparisons)
                atomic_json(out/'release-decision.json',{'accepted':False,'reason':'postrun_audits_pending','weights_public':False})
            stage('completed',complete=True,finished_utc=utc(),finished_epoch=time.time())
        except BaseException as error:
            stage('failed',error_type=type(error).__name__,finished_utc=utc(),finished_epoch=time.time());raise
        finally:
            state['telemetry']=monitor.stop();atomic_json(out/'controller.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();execute(a.root,a.out,a.resume)
