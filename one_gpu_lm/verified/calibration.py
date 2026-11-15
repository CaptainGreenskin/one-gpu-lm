"""One preregistered submission-guard calibration on the same training-only tasks."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from one_gpu_lm.common.receipts import sha256,write_json
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.serving.client import request
from .sandbox import Sandbox
from .tasks import validate_source,validate_split
from .rollout import run as rollout,atomic
from .submission import SubmissionGuard


def run(root,protocol_path):
    root=Path(root).resolve();protocol_path=Path(protocol_path).resolve();protocol=json.loads(protocol_path.read_text())
    out=root/'runs/teacher-calibration-001'
    if out.exists():raise ValueError('A pilot attempt already exists; no duplicate launch')
    publication=json.loads((root/'private/calibration-preregistration.json').read_text())
    if publication['anonymous_files_verified'] is not True or publication['ci_conclusion']!='success' or publication['protocol_sha256']!=sha256(protocol_path):
        raise ValueError('Published pilot protocol not verified')
    for name,expected in protocol['frozen_files'].items():
        if sha256(root/name)!=expected:raise ValueError('Frozen input changed: '+name)
    if protocol['collection_policy'] != 'explicit_submission_guard_v1':raise ValueError('Unknown calibration policy')
    previous=json.loads((root/'runs/teacher-pilot-001/controller.json').read_text())
    if not previous['complete'] or not previous['backend_stopped']:raise ValueError('Previous pilot must be finished')
    if publication['previous_result_commit']!=protocol['previous_result_commit'] or publication['previous_result_public_verified'] is not True:raise ValueError('Previous results must be public and verified')
    admission=json.loads((root/protocol['admission_path']).read_text())
    if not admission['complete'] or not admission['passed']:raise ValueError('CPU admission incomplete')
    data=root/'private/pilot-data';source_rows=json.loads((data/'sources.json').read_text())['sources'];tasks=json.loads((data/'tasks.json').read_text())['tasks']
    if len(tasks)!=12 or [t['id'] for t in tasks]!=protocol['task_ids'] or any(t['split']!='training' for t in tasks):
        raise ValueError('Unexpected task scope/order')
    if admission['tasks_sha256']!=sha256(data/'tasks.json') or admission['sources_sha256']!=sha256(data/'sources.json'):
        raise ValueError('Tasks or rights metadata changed since admission')
    validate_split(source_rows)
    sources={s['repository']:s for s in source_rows}
    for source in source_rows:validate_source(source,data/'snapshots'/source['repository'].replace('/','--'),'training')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise ValueError('GPU already occupied; no competing launch')
    runtime=json.loads((root/'private/parent-runtime.json').read_text())
    sandbox=Sandbox(json.loads((root/'private/task-runtime.json').read_text()),protocol['teacher_budget']['max_tool_seconds'])
    if admission['runtime_sha256']!=sha256(root/'private/task-runtime.json'):raise ValueError('CPU runtime differs from admission')
    sandbox=SubmissionGuard(sandbox,{str(out/'private/tasks'/t['id']/'workspace'):{k:t[k] for k in ['editable_files','regression_command']} for t in tasks})
    out.mkdir();(out/'private').mkdir()
    started=time.time();deadline=started+protocol['max_pipeline_seconds'];backend=None
    state={'schema_version':1,'stage':'verify_model','complete':False,'started_epoch':started,
           'started_utc':datetime.now(timezone.utc).isoformat(),'deadline_epoch':deadline,'tasks':[],
           'protocol_sha256':sha256(protocol_path),'final_test_opened':False,'development_tasks_scored':0,'training_updates':0,
           'external_teacher_requests':0,'publication_commit':publication['commit'],
           'collection_policy':protocol['collection_policy'],'previous_result_commit':protocol['previous_result_commit']}
    def save():atomic(out/'controller.json',state)
    save();telemetry=Telemetry(out/'telemetry.csv',1);telemetry.start()
    try:
        profile_path=root/'experiments/025-verified-distillation/teacher-profile.json';profile=json.loads(profile_path.read_text())
        verification=verify_model(Path(runtime['model']),root/'experiments/011-coding-model-selection/weights-manifest.json',profile)
        write_json(out/'model-verification.json',verification)
        if sha256(Path(runtime['model'])/'LICENSE')!=protocol['teacher_rights']['license_sha256']:
            raise ValueError('Teacher license changed')
        state['stage']='loading_teacher';save()
        socket=out/'backend.sock';backend_out=out/'private/backend'
        log=(out/'private/backend.log').open('wb')
        backend=subprocess.Popen([runtime['model_python'],'-m','one_gpu_lm.serving.local_backend','--model',runtime['model'],
            '--profile',str(profile_path),'--socket',str(socket),'--out',str(backend_out)],cwd=root,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['backend_pid']=backend.pid;save();load_deadline=min(deadline,time.time()+600)
        while not (backend_out/'ready.json').exists():
            if backend.poll() is not None:raise RuntimeError('Teacher backend exited before readiness')
            if time.time()>=load_deadline:raise TimeoutError('Teacher loading exceeded cap')
            time.sleep(.5)
        health=request(socket,{'operation':'health'});write_json(out/'served-artifact.json',health)
        for task in tasks:
            if time.time()+protocol['teacher_budget']['max_task_seconds']+60>deadline:
                state['stage']='pipeline_budget_exhausted';break
            state.update(stage='collecting',current_task=task['id']);save()
            result=rollout(root,data,task,sources[task['repository']],socket,sandbox,protocol,out/'private/tasks'/task['id'])
            trace=json.loads((out/'private/tasks'/task['id']/'trajectory.json').read_text())
            result['collection_policy']=protocol['collection_policy']
            result['rejected_unexecuted_requests']=sum(s.get('observation',{}).get('executed') is False for s in trace['steps'])
            write_json(out/'private/tasks'/task['id']/'result.json',result)
            state['tasks'].append(result);save()
            print(json.dumps({'task':task['id'],'passed':result['grade']['passed'],'eligible':result['eligible_verified_trajectory'],
                              'calls':result['attempted_calls'],'generated_tokens':result['generated_tokens']}),flush=True)
        if len(state['tasks'])==len(tasks):state.update(complete=True,stage='completed')
    except Exception as error:
        state.update(stage='failed',error_type=type(error).__name__)
        (out/'private/controller-error.log').write_text(traceback.format_exc())
    finally:
        if backend is not None and backend.poll() is None:
            os.killpg(backend.pid,signal.SIGTERM)
            try:backend.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(backend.pid,signal.SIGKILL);backend.wait(timeout=10)
        state['telemetry']=telemetry.stop()
        state.update(finished_epoch=time.time(),finished_utc=datetime.now(timezone.utc).isoformat(),backend_stopped=backend is None or backend.poll() is not None)
        state['frozen_files_unchanged']=all(sha256(root/n)==h for n,h in protocol['frozen_files'].items())
        save()
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    a=p.parse_args();run(a.root,a.protocol)
