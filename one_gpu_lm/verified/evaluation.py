"""One preregistered 720-run development comparison, with no implicit retries."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.serving.client import request
from .evaluation_rollout import run as rollout
from .factorial import ARMS, schedule
from .rollout import atomic
from .sandbox import Sandbox
from .submission import SubmissionGuard
from .tasks import relative, validate_source, validate_split


def validate_scope(protocol, spec, tasks, sources, training_sources, previous_families, admission,
                   tasks_hash, sources_hash, runtime_hash):
    dev=spec['development'];ids=[t['id'] for t in tasks]
    if (not tasks or len(set(ids))!=len(ids) or len(tasks)!=dev['unique_tasks']
            or any(t['split']!='development' for t in tasks)):
        raise ValueError('Unexpected development-only task inventory')
    if protocol['evaluation_budget']!=dev['common_budget'] or dev['arms']!=ARMS or dev['repetitions']!=3:
        raise ValueError('Changed arm, repetition or common-budget rules')
    if protocol['schedule']!=schedule(ids,dev['repetitions']) or len(protocol['schedule'])!=dev['total_task_runs']:
        raise ValueError('Changed registered task/arm order')
    if (tasks_hash!=dev['tasks_sha256'] or sources_hash!=dev['sources_sha256']
            or admission['tasks_sha256']!=tasks_hash or admission['sources_sha256']!=sources_hash
            or admission['runtime_sha256']!=runtime_hash or not admission['complete'] or not admission['passed']):
        raise ValueError('Development admission or artifact identity mismatch')
    if [t['id'] for t in admission['tasks'] if t['admitted']]!=ids or len(admission['tasks'])!=len(ids):
        raise ValueError('Missing, failed or reordered admitted task')
    validate_split([*training_sources,*sources],previous_families)
    by_repo={s['repository']:s for s in sources};families=Counter(t['family'] for t in tasks)
    if (len(families)!=dev['repository_families'] or set(families.values())!={10}
            or len({s['language'] for s in sources})!=dev['languages']
            or {t['repository'] for t in tasks}!=set(by_repo)):
        raise ValueError('Registered family/language coverage differs')
    for task in tasks:
        source=by_repo[task['repository']]
        if (source['split']!='development' or source['approved_for'].get('development') is not True
                or task['family']!=source['family'] or task['revision']!=source['revision']):
            raise ValueError('Unapproved or mismatched development source')


def validate_candidate(candidate, supervisor, expected_protocol_hash):
    flags=['complete','frozen_files_unchanged','saved_reload_greedy_ids_equal','all_selected_records_consumed',
           'frozen_sample_unchanged','frozen_gradients_absent','adapter_changed']
    if (not all(candidate.get(k) is True for k in flags)
            or candidate['protocol_sha256']!=expected_protocol_hash
            or candidate.get('optimizer_steps',0)<1 or candidate.get('candidate_count')!=1
            or not supervisor['complete'] or not supervisor['candidate_eligible']
            or supervisor['timed_out'] or supervisor['worker_returncode']!=0):
        raise ValueError('Only the complete, verified single candidate may enter development evaluation')


class Backend:
    """Reload only when changing weights; all reload cost remains in pipeline time."""
    def __init__(self, root, out, runtime, protocol, record, changed=lambda:None):
        self.root,self.out,self.runtime,self.protocol=Path(root),Path(out),runtime,protocol
        self.record=record;self.process=None;self.role=None;self.number=0;self.entry=None
        self.changed=changed
        self.socket=self.out/'backend.sock'
    def check_exclusive(self):
        pids={int(p) for p in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()}
        if self.process is None or pids!={self.process.pid}:
            raise RuntimeError('GPU compute ownership changed; preserve the run instead of collecting confounded scores')
    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                os.killpg(self.process.pid,signal.SIGTERM)
                try:self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid,signal.SIGKILL);self.process.wait(timeout=10)
            if self.entry is not None:
                self.entry.update(stopped_epoch=time.time(),returncode=self.process.returncode)
            self.process=None;self.role=None
            self.changed()
        if self.socket.exists():self.socket.unlink()
    def ensure(self, role, deadline):
        if role not in {'base','student'}:raise ValueError('Unexpected weight role')
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError('Unexpected backend exit; do not silently reload and continue')
        if self.role==role and self.process is not None and self.process.poll() is None:
            self.check_exclusive()
            return self.socket
        self.stop()
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            raise ValueError('GPU occupied by another process; no competing load')
        self.number+=1;directory=self.out/'private/backends'/f'{self.number:04d}-{role}'
        directory.parent.mkdir(parents=True,exist_ok=True)
        args=[self.runtime['model_python'],'-m','one_gpu_lm.verified.evaluation_backend',
              '--expected-environment',str(self.root/relative(self.protocol['environment_path'])),
              '--environment-receipt',str(directory.parent/f'{self.number:04d}-{role}-environment.json'),
              '--model',self.runtime['model'],
              '--profile',str(self.root/relative(self.protocol['evaluation_profile_path'])),
              '--socket',str(self.socket),'--out',str(directory)]
        if role=='student':
            adapter=self.root/relative(self.protocol['adapter_directory'])
            if {p.name:sha256(p) for p in adapter.iterdir() if p.is_file()}!=self.protocol['all_adapter_files_sha256']:
                raise ValueError('Candidate adapter changed before loading')
            args+=['--adapter',str(adapter)]
        self.entry={'index':self.number,'role':role,'started_epoch':time.time()}
        self.record.append(self.entry)
        self.process=subprocess.Popen(args,cwd=self.root,stdout=(directory.parent/f'{self.number:04d}-{role}.log').open('wb'),
                    stderr=subprocess.STDOUT,start_new_session=True)
        self.role=role;self.entry['pid']=self.process.pid
        self.changed()
        load_deadline=min(deadline,time.time()+600)
        while not (directory/'ready.json').exists():
            if self.process.poll() is not None:raise RuntimeError('Evaluation backend exited before readiness')
            if time.time()>=load_deadline:raise TimeoutError('Evaluation backend load cap exceeded')
            time.sleep(.5)
        self.entry['ready_epoch']=time.time()
        write_json(directory/'served-artifact.json',request(self.socket,{'operation':'health'}))
        self.check_exclusive()
        return self.socket


def run(root, protocol_path):
    root,protocol_path=Path(root).resolve(),Path(protocol_path).resolve()
    protocol=json.loads(protocol_path.read_text());out=root/relative(protocol['run_directory'])
    if out.exists():raise ValueError('Evaluation already attempted; do not duplicate or overwrite')
    publication=json.loads((root/relative(protocol['publication_receipt'])).read_text())
    if (publication['anonymous_files_verified'] is not True or publication['ci_conclusion']!='success'
            or publication['protocol_sha256']!=sha256(protocol_path)):
        raise ValueError('Executable evaluation preregistration must be publicly verified')
    for name,expected in protocol['frozen_files'].items():
        if sha256(root/relative(name))!=expected:raise ValueError('Frozen evaluation input changed: '+name)
    spec=json.loads((root/relative(protocol['cycle_spec'])).read_text())
    data=root/relative(protocol['data_directory'])
    tasks=json.loads((data/'tasks.json').read_text())['tasks']
    sources=json.loads((data/'sources.json').read_text())['sources']
    train_sources=json.loads((root/relative(protocol['training_sources_path'])).read_text())['sources']
    previous=json.loads((root/relative(protocol['previous_families_path'])).read_text())['repository_names']
    admission=json.loads((root/relative(protocol['admission_path'])).read_text())
    validate_scope(protocol,spec,tasks,sources,train_sources,previous,admission,
                   sha256(data/'tasks.json'),sha256(data/'sources.json'),sha256(root/'private/task-runtime.json'))
    candidate=json.loads((root/relative(protocol['candidate_result'])).read_text())
    candidate_supervisor=json.loads((root/relative(protocol['candidate_supervisor'])).read_text())
    validate_candidate(candidate,candidate_supervisor,sha256(root/relative(protocol['candidate_protocol'])))
    candidate_plan=json.loads((root/relative(protocol['candidate_protocol'])).read_text())
    if (candidate['profile_sha256']!=sha256(root/relative(candidate_plan['profile_path']))
            or sha256(root/relative(candidate_plan['cycle_spec']))!=sha256(root/relative(protocol['cycle_spec']))
            or candidate['expected_environment_sha256']!=sha256(root/relative(protocol['environment_path']))):
        raise ValueError('Candidate profile or scientific rules changed since training')
    for name,expected in candidate_plan['frozen_files'].items():
        if sha256(root/relative(name))!=expected:raise ValueError('Candidate provenance input changed: '+name)
    for source in sources:validate_source(source,data/'snapshots'/source['repository'].replace('/','--'),'development')
    runtime=json.loads((root/'private/parent-runtime.json').read_text())
    sandbox=Sandbox(json.loads((root/'private/task-runtime.json').read_text()),protocol['evaluation_budget']['max_tool_seconds'])
    guarded=SubmissionGuard(sandbox,{str(out/'private/tasks'/row['run_id']/'workspace'):
       {k:next(t for t in tasks if t['id']==row['task_id'])[k] for k in ['editable_files','regression_command']}
       for row in protocol['schedule'] if row['arm'].endswith('_guard')})
    out.mkdir(parents=True);(out/'private').mkdir()
    started=time.time();deadline=started+spec['development']['maximum_evaluation_pipeline_seconds']
    state={'schema_version':1,'stage':'verify_model','complete':False,'started_epoch':started,
           'started_utc':datetime.now(timezone.utc).isoformat(),'deadline_epoch':deadline,'runs':[],
           'protocol_sha256':sha256(protocol_path),'publication_commit':publication['commit'],
           'backend_lifecycles':[],'expected_runs':len(protocol['schedule']),
           'optimizer_updates_during_evaluation':0,'final_test_opened':False,'weights_public':False}
    def save():atomic(out/'controller.json',state)
    save();telemetry=Telemetry(out/'telemetry.csv',1);telemetry.start()
    backend=Backend(root,out,runtime,protocol,state['backend_lifecycles'],save)
    try:
        profile_path=root/relative(protocol['evaluation_profile_path']);profile=json.loads(profile_path.read_text())
        if sha256(profile_path)!=protocol['evaluation_profile_sha256']:raise ValueError('Evaluation profile hash differs')
        write_json(out/'model-verification.json',verify_model(Path(runtime['model']),root/relative(protocol['model_manifest_path']),profile))
        by_id={t['id']:t for t in tasks};by_repo={s['repository']:s for s in sources}
        for item in protocol['schedule']:
            role='student' if item['arm'].startswith('student_') else 'base'
            reserve=protocol['evaluation_budget']['max_task_seconds']+120+(600 if backend.role!=role else 0)
            if time.time()+reserve>deadline:
                state['stage']='pipeline_budget_exhausted';break
            state.update(stage='loading_backend',current_run=item);save()
            socket=backend.ensure(role,deadline);state.update(stage='evaluating');save()
            task=by_id[item['task_id']]
            value=rollout(root,data,task,by_repo[task['repository']],socket,
                 guarded if item['arm'].endswith('_guard') else sandbox,protocol,out/'private/tasks'/item['run_id'],item['arm'])
            value.update(repetition=item['repetition'],run_id=item['run_id'])
            trace=json.loads((out/'private/tasks'/item['run_id']/'trajectory.json').read_text())
            value.update(input_tokens=sum(s['generation']['input_tokens'] for s in trace['steps']),
                         rejected_unexecuted_requests=sum(s.get('observation',{}).get('executed') is False for s in trace['steps']))
            write_json(out/'private/tasks'/item['run_id']/'result.json',value)
            state['runs'].append(value);save()
            print(json.dumps({'run_id':item['run_id'],'success_within_budget':value['success_within_budget'],
                              'calls':value['attempted_calls'],'generated_tokens':value['generated_tokens']}),flush=True)
        if len(state['runs'])==len(protocol['schedule']):state.update(stage='completed',complete=True)
    except BaseException as error:
        state.update(stage='failed',error_type=type(error).__name__)
        (out/'private/error.log').write_text(traceback.format_exc())
    finally:
        backend.stop();state['telemetry']=telemetry.stop()
        state.update(finished_epoch=time.time(),finished_utc=datetime.now(timezone.utc).isoformat(),backend_stopped=True,
          frozen_files_unchanged=all((root/relative(n)).is_file() and sha256(root/relative(n))==h for n,h in protocol['frozen_files'].items()))
        if not state['frozen_files_unchanged']:state.update(complete=False,stage='frozen_input_changed')
        save()
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    a=p.parse_args();run(a.root,a.protocol)
