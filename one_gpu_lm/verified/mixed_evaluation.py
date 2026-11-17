"""One preregistered three-checkpoint development campaign; no final-test access."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
from one_gpu_lm.common.receipts import sha256,write_json
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.verify import verify_model
from .evaluation import Backend,validate_candidate
from .evaluation_rollout import run as rollout,atomic
from .mixed_eval_design import VARIANTS,schedule,effective_protocol,internal_arm
from .sandbox import Sandbox
from .tasks import relative,validate_source,validate_split


def validate_scope(p,spec,tasks,sources,training_sources,admission,tasks_hash,sources_hash,runtime_hash):
    dev=spec['development'];ids=[t['id'] for t in tasks];families=Counter(t['family'] for t in tasks)
    if (len(ids)!=60 or len(set(ids))!=60 or any(t['split']!='development' for t in tasks)
            or dev['variants']!=VARIANTS or dev['repetitions']!=3 or dev['block_size']!=10
            or dev['total_task_runs']!=540 or p['evaluation_budget']!=dev['common_budget']
            or p['schedule']!=schedule(ids,3,10) or len(families)!=6 or set(families.values())!={10}):
        raise ValueError('Registered development-only scope changed')
    if (tasks_hash!=dev['tasks_sha256'] or sources_hash!=dev['sources_sha256']
            or not admission['complete'] or not admission['passed'] or admission['tasks_sha256']!=tasks_hash
            or admission['sources_sha256']!=sources_hash or admission['runtime_sha256']!=runtime_hash
            or [t['id'] for t in admission['tasks'] if t['admitted']]!=ids or len(admission['tasks'])!=60):
        raise ValueError('Frozen admission identity mismatch')
    validate_split([*training_sources,*sources])
    by={s['repository']:s for s in sources}
    if (set(by)!={t['repository'] for t in tasks} or len({s['language'] for s in sources})!=2
            or {s['family'] for s in sources}&{s['family'] for s in training_sources}):
        raise ValueError('Training overlap or development coverage mismatch')
    for t in tasks:
        s=by[t['repository']]
        if s['split']!='development' or s['approved_for'].get('development') is not True or s['family']!=t['family'] or s['revision']!=t['revision']:
            raise ValueError('Development task provenance mismatch')


def run(root,protocol_path):
    root,protocol_path=Path(root).resolve(),Path(protocol_path).resolve();p=json.loads(protocol_path.read_text());out=root/relative(p['run_directory'])
    if out.exists():raise ValueError('Preserve the original development attempt')
    read=lambda n:json.loads((root/relative(n)).read_text())
    public=read(p['publication_receipt'])
    if not public['anonymous_files_verified'] or public['ci_conclusion']!='success' or public['protocol_sha256']!=sha256(protocol_path):raise ValueError('Public exact-commit CI and protocol required')
    for n,h in p['frozen_files'].items():
        if sha256(root/relative(n))!=h:raise ValueError('Frozen input changed: '+n)
    spec=read(p['cycle_spec']);data=root/relative(p['data_directory']);tasks=read(p['data_directory']+'/tasks.json')['tasks'];sources=read(p['data_directory']+'/sources.json')['sources']
    validate_scope(p,spec,tasks,sources,read(p['training_sources_path'])['sources'],read(p['admission_path']),sha256(data/'tasks.json'),sha256(data/'sources.json'),sha256(root/'private/task-runtime.json'))
    if p['variants']['original025']['adapter_sha256']==p['variants']['mixed031']['adapter_sha256']:
        raise ValueError('Distinct candidate weight identities required')
    for variant in VARIANTS:
        effective_protocol(p,variant)
        if variant=='base':continue
        model=p['variants'][variant];candidate=read(model['candidate_result']);supervisor=read(model['candidate_supervisor']);plan=read(model['candidate_protocol'])
        validate_candidate(candidate,supervisor,sha256(root/relative(model['candidate_protocol'])))
        if candidate['profile_sha256']!=sha256(root/relative(plan['profile_path'])) or candidate['expected_environment_sha256']!=sha256(root/relative(p['environment_path'])):
            raise ValueError('Candidate training identity mismatch')
        if variant=='mixed031':
            if sha256(root/relative(plan['cycle_spec']))!=sha256(root/relative(p['cycle_spec'])):raise ValueError('Scientific rules changed after training')
            for n,h in plan['frozen_files'].items():
                if sha256(root/relative(n))!=h:raise ValueError('Mixed training provenance input changed: '+n)
            if not read(p['training_integrity_path'])['passed']:raise ValueError('Mixed training integrity audit required')
    for source in sources:validate_source(source,data/'snapshots'/source['repository'].replace('/','--'),'development')
    runtime=read('private/parent-runtime.json');runner=Sandbox(read('private/task-runtime.json'),p['evaluation_budget']['max_tool_seconds'])
    out.mkdir(parents=True);(out/'private').mkdir();started=time.time();deadline=started+spec['development']['maximum_evaluation_pipeline_seconds']
    state={'schema_version':1,'stage':'verify_model','complete':False,'started_epoch':started,'started_utc':datetime.now(timezone.utc).isoformat(),
           'deadline_epoch':deadline,'runs':[],'backend_lifecycles':[],'protocol_sha256':sha256(protocol_path),'publication_commit':public['commit'],
           'expected_runs':len(p['schedule']),'optimizer_updates_during_evaluation':0,'final_test_opened':False,'weights_public':False}
    def save():atomic(out/'controller.json',state)
    save();telemetry=Telemetry(out/'telemetry.csv',1);telemetry.start();backend=Backend(root,out,runtime,p,state['backend_lifecycles'],save);previous=None
    try:
        profile=read(p['evaluation_profile_path'])
        if profile['chat_template_kwargs']!={'enable_thinking':False} or profile.get('generation_modes') or sha256(root/relative(p['evaluation_profile_path']))!=p['evaluation_profile_sha256']:
            raise ValueError('All variants must use the same non-thinking profile')
        write_json(out/'model-verification.json',verify_model(Path(runtime['model']),root/relative(p['model_manifest_path']),profile))
        by_id={t['id']:t for t in tasks};by_repo={s['repository']:s for s in sources}
        for item in p['schedule']:
            variant=item['variant'];effective=effective_protocol(p,variant);role='base' if variant=='base' else 'student'
            if time.time()+p['evaluation_budget']['max_task_seconds']+120+(600 if previous!=variant else 0)>deadline:
                state['stage']='pipeline_budget_exhausted';break
            state.update(stage='loading_backend',current_run=item);save()
            if previous!=variant:backend.stop()
            backend.protocol=effective;socket=backend.ensure(role,deadline);backend.entry['variant']=variant;previous=variant
            state['stage']='evaluating';save();task=by_id[item['task_id']];directory=out/'private/tasks'/item['run_id']
            value=rollout(root,data,task,by_repo[task['repository']],socket,runner,effective,directory,internal_arm(variant));backend.check_exclusive()
            trace=json.loads((directory/'trajectory.json').read_text())
            value.update(variant=variant,repetition=item['repetition'],block=item['block'],run_id=item['run_id'],
              input_tokens=sum(s['generation']['input_tokens'] for s in trace['steps']),rejected_unexecuted_requests=0)
            write_json(directory/'result.json',value);state['runs'].append(value);save()
            print(json.dumps({'completed_runs':len(state['runs']),'run_id':item['run_id'],'success':value['success_within_budget']}),flush=True)
        if len(state['runs'])==len(p['schedule']):state.update(stage='completed',complete=True)
    except BaseException as error:
        state.update(stage='failed',error_type=type(error).__name__);(out/'private/error.log').write_text(traceback.format_exc())
    finally:
        backend.stop();state['telemetry']=telemetry.stop();state.update(finished_epoch=time.time(),finished_utc=datetime.now(timezone.utc).isoformat(),backend_stopped=True,
          frozen_files_unchanged=all(sha256(root/relative(n))==h for n,h in p['frozen_files'].items()))
        if not state['frozen_files_unchanged']:state.update(complete=False,stage='frozen_input_changed')
        save()
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);a=p.parse_args();run(a.root,a.protocol)
