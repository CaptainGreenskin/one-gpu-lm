"""Resume the exact frozen schedule after an independently observed host restart."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback
from one_gpu_lm.common.receipts import sha256,write_json
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.verify import verify_model
from .evaluation import validate_candidate
from .historic_evaluation import validate_scope
from .resilient_backend_control import JournalBackend
from .historic_rollout import atomic
from .resilient_rollout import run as rollout
from .historic_eval_design import VARIANTS,effective_protocol,internal_arm
from .sandbox import Sandbox
from .tasks import relative,validate_source
from .historic_tasks import adapt
from .resume_prefix import restore_prefix


def validate_repeat(root, protocol):
    parent = root/relative(protocol['original_protocol_path'])
    if sha256(parent) != protocol['original_protocol_sha256']:
        raise ValueError('Original evaluation protocol changed')
    original = json.loads(parent.read_text())
    new_attempt_keys = {'stage','run_directory','publication_receipt','supervisor_receipt','worker_log',
                        'frozen_files','new_development_scores_before_freeze'}
    if any(original[key] != protocol[key] for key in original.keys() - new_attempt_keys):
        raise ValueError('Model, data, budgets, schedule, or grading differ from original evaluation')
    if any(protocol['frozen_files'].get(name) != digest for name,digest in original['frozen_files'].items()):
        raise ValueError('Original frozen files must still match')
    smoke = root/relative(protocol['smoke_receipt_path'])
    if sha256(smoke) != protocol['smoke_receipt_sha256']:
        raise ValueError('GPU smoke receipt changed')
    receipt = json.loads(smoke.read_text())
    if (not receipt['passed'] or receipt['final_test_opened'] or not receipt['backend_stopped']
            or len(receipt['tasks']) != 2 or not all(all(row['checks'].values()) for row in receipt['tasks'])
            or receipt['adapter_sha256'] != protocol['variants']['multifile032']['adapter_sha256']):
        raise ValueError('Passing GPU timeout and continuation required')
    if protocol['timeout_recovery_seconds'] != 180:
        raise ValueError('Changed post-deadline accounting window')


def run(root,protocol_path):
    root,protocol_path=Path(root).resolve(),Path(protocol_path).resolve();p=json.loads(protocol_path.read_text());out=root/relative(p['run_directory'])
    if out.exists():raise ValueError('Preserve the original development attempt')
    read=lambda n:json.loads((root/relative(n)).read_text())
    public=read(p['publication_receipt'])
    if not public['anonymous_files_verified'] or public['ci_conclusion']!='success' or public['protocol_sha256']!=sha256(protocol_path):raise ValueError('Public exact-commit CI and protocol required')
    for n,h in p['frozen_files'].items():
        if sha256(root/relative(n))!=h:raise ValueError('Frozen input changed: '+n)
    validate_repeat(root,p)
    spec=read(p['cycle_spec']);data=root/relative(p['data_directory']);tasks=read(p['data_directory']+'/tasks.json')['tasks'];sources=read(p['data_directory']+'/sources.json')['sources']
    validate_scope(p,spec,tasks,sources,read(p['training_sources_path'])['sources'],read(p['admission_path']),sha256(data/'tasks.json'),sha256(data/'sources.json'),{r:sha256(Path(path)) for r,path in read(p['runtime_map']).items()})
    if p['variants']['mixed031']['adapter_sha256']==p['variants']['multifile032']['adapter_sha256']:
        raise ValueError('Distinct candidate weight identities required')
    for variant in VARIANTS:
        effective_protocol(p,variant)
        if variant=='base':continue
        model=p['variants'][variant];candidate=read(model['candidate_result']);supervisor=read(model['candidate_supervisor']);plan=read(model['candidate_protocol'])
        validate_candidate(candidate,supervisor,sha256(root/relative(model['candidate_protocol'])))
        if candidate['profile_sha256']!=sha256(root/relative(plan['profile_path'])) or candidate['expected_environment_sha256']!=sha256(root/relative(p['environment_path'])):
            raise ValueError('Candidate training identity mismatch')
        if variant=='multifile032':
            if sha256(root/relative(plan['cycle_spec']))!=sha256(root/relative(p['cycle_spec'])):raise ValueError('Scientific rules changed after training')
            for n,h in plan['frozen_files'].items():
                if sha256(root/relative(n))!=h:raise ValueError('Mixed training provenance input changed: '+n)
            if not read(p['training_integrity_path'])['passed']:raise ValueError('Mixed training integrity audit required')
    source_by_id={s['source_id']:s for s in sources}
    for task in tasks:
        source=source_by_id[task['source_id']];adapted,_=adapt(source,task)
        validate_source(adapted,data/'snapshots'/source['source_id'],'development')
    runtime=read('private/parent-runtime.json');runners={repo:Sandbox(json.loads(Path(path).read_text()),p['evaluation_budget']['max_tool_seconds']) for repo,path in read(p['runtime_map']).items()}
    recovery=read(p['recovery_receipt_path'])
    restored=restore_prefix(root,out,p,recovery)
    started=time.time();deadline=restored['deadline_epoch']
    state=dict(restored,stage='verify_model',complete=False,resumed_epoch=started,
               preparation_seconds=started-restored['resume_preparation_started_epoch'],
               protocol_sha256=sha256(protocol_path),publication_commit=public['commit'])
    prefix_count=len(state['runs'])
    def save():atomic(out/'controller.json',state)
    save();telemetry=Telemetry(out/'telemetry.csv',2);telemetry.start();backend=JournalBackend(root,out,runtime,p,state['backend_lifecycles'],save);previous=None
    backend.number=len(state['backend_lifecycles'])
    try:
        profile=read(p['evaluation_profile_path'])
        if profile['chat_template_kwargs']!={'enable_thinking':False} or profile.get('generation_modes') or sha256(root/relative(p['evaluation_profile_path']))!=p['evaluation_profile_sha256']:
            raise ValueError('All variants must use the same non-thinking profile')
        write_json(out/'model-verification.json',verify_model(Path(runtime['model']),root/relative(p['model_manifest_path']),profile))
        by_id={t['id']:t for t in tasks};by_repo={s['source_id']:s for s in sources}
        for item in p['schedule'][prefix_count:]:
            variant=item['variant'];effective=effective_protocol(p,variant);role='base' if variant=='base' else 'student'
            if time.time()+p['evaluation_budget']['max_task_seconds']+p['timeout_recovery_seconds']+120+(600 if previous!=variant else 0)>deadline:
                state['stage']='pipeline_budget_exhausted';break
            state.update(stage='loading_backend',current_run=item);save()
            if previous!=variant:backend.stop()
            backend.protocol=effective;socket=backend.ensure(role,deadline);backend.entry['variant']=variant;previous=variant
            state['stage']='evaluating';save();task=by_id[item['task_id']];directory=out/'private/tasks'/item['run_id']
            value=rollout(root,data,task,by_repo[task['source_id']],socket,runners[task['repository']],effective,
                          directory,internal_arm(variant),journal_path=backend.journal_path,
                          recovery_seconds=p['timeout_recovery_seconds'])
            backend.check_exclusive()
            trace=json.loads((directory/'trajectory.json').read_text())
            value.update(variant=variant,repetition=item['repetition'],block=item['block'],run_id=item['run_id'],
              input_tokens=sum(s['generation']['input_tokens'] for s in trace['steps']),rejected_unexecuted_requests=0,backend_index=backend.number)
            write_json(directory/'result.json',value);state['runs'].append(value);save();os.sync()
            print(json.dumps({'completed_runs':len(state['runs']),'run_id':item['run_id'],'success':value['complete_success_within_budget']}),flush=True)
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
