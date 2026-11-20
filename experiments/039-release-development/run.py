"""One registered 27B base and LoRA comparison using the same original prefix."""
import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_tasks import adapt, validate_split
from one_gpu_lm.verified.resilient_backend_control import JournalBackend
from one_gpu_lm.verified.resilient_rollout import run as rollout
from one_gpu_lm.verified.rollout import atomic, normalization
from one_gpu_lm.verified.sandbox import Sandbox
from one_gpu_lm.verified.tasks import relative, validate_source

from analysis import schedule
from recovery import AMENDMENT, EXP, RESUMED, inspect, restore, served_weight_hashes, validate_amendment


REFERENCE = Path('experiments/038-independent-compact')
PRIVATE = Path('private/modelscope-development')
DATA = PRIVATE/'development'
OUT = Path('runs/q039-development')
TEST_RUNTIME = PRIVATE/'offline-runtime-v2.json'
DATE_RUNTIME = PRIVATE/'offline-runtime-date.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def frozen(root, files):
    return all((root/relative(name)).is_file() and sha256(root/relative(name))==value for name,value in files.items())


def test_runtime(family):
    return DATE_RUNTIME if family=='date-fns/date-fns' else TEST_RUNTIME


def effective(common, arm):
    if arm=='base':
        return dict(common,evaluation_variant='base',adapter_sha256={},adapter_directory=None,all_adapter_files_sha256={})
    if arm=='candidate':
        return dict(common,evaluation_variant='candidate')
    raise ValueError('Unsupported checkpoint arm')


@dataclass
class Context:
    plan: dict
    plan_path: Path
    common: dict
    tasks: dict
    sources: dict
    prompts: dict
    proof: dict
    order: list
    runtime: dict
    files: dict
    recovery: dict = None


def reference_protocol(root):
    parent=read(root/'experiments/032-multifile-completion/evaluation-protocol.json')
    older=read(root/REFERENCE/'protocol.json')
    keys=('model_id','revision','evaluation_budget','agent_policy','normalization')
    if any(older[key]!=parent[key] for key in keys):
        raise ValueError('Previous runtime conditioning or model changed')
    common={k:parent[k] for k in (*keys, 'environment_path','evaluation_profile_path','evaluation_profile_sha256','model_manifest_path')}
    return dict(common,final_test_opened=False)


def prefix_prompt(root, common, task, output):
    evidence=load('historic036_evidence',root/'experiments/036-regression-evidence/evidence.py')
    return evidence.make_task(task,output,'transcript',normalization(root),common['evaluation_budget']['max_tool_output_chars'])


def inputs(root, *, resume=False):
    path=root/EXP/'protocol.json'
    plan=read(path)
    design=read(root/EXP/'design.json')
    recovery, files = (validate_amendment(root,path,plan,root/PRIVATE/'recovery-publication.json')
                       if resume else (None,plan['frozen_files']))
    if any(plan.get(k)!=v for k,v in design.items()) or not frozen(root,files):
        raise ValueError('Changed scientific design or frozen input')
    publication=read(root/PRIVATE/'publication.json')
    if (not publication['anonymous_files_verified'] or publication['ci_conclusion']!='success'
            or publication['protocol_sha256']!=sha256(path)):
        raise ValueError('Expected the public immutable preregistration and CI')
    common=reference_protocol(root)
    if (common['model_id']!=design['pretrained_model'] or common['revision']!=design['base_revision']
            or common['evaluation_budget']!=design['evaluation_budget']):
        raise ValueError('038 prefix and original model budget required')
    candidate=read(root/PRIVATE/'candidate-input-review.json')
    adapter_dir=relative(plan['adapter_directory'])
    actual={p.name:sha256(p) for p in (root/adapter_dir).iterdir() if p.is_file()}
    if (not candidate['audited_checkpoint_matches'] or not candidate['eligible_for_existing_scope_of_development']
            or candidate['model_id']!=common['model_id'] or candidate['revision']!=common['revision']
            or candidate['checkpoint_files_sha256']!=actual or plan['all_adapter_files_sha256']!=actual):
        raise ValueError('031 candidate differs from the reviewed weight files')
    common.update(adapter_directory=str(adapter_dir),all_adapter_files_sha256=actual,
                  adapter_sha256=served_weight_hashes(actual))
    tasks={i:t for i,t in enumerate(read(root/DATA/'tasks.json')['tasks'],1)}
    sources={s['source_id']:s for s in read(root/DATA/'sources.json')['sources']}
    proof=read(root/DATA/'preflight.json')
    if (not proof['passed'] or proof['model_calls'] or proof['final_test_opened'] or len(proof['rows'])!=len(tasks)
            or Counter(t['family'] for t in tasks.values())!=plan['families']
            or Counter(s['language'] for s in sources.values())!=plan['language_counts']):
        raise ValueError('Only the CPU-verified independent development cohort is allowed')
    validate_split(list(sources.values()),proof['prior_families'])
    prompts={}
    for number,task in tasks.items():
        source=sources[task['source_id']]
        adapted,_=adapt(source,task)
        validate_source(adapted,root/DATA/'snapshots'/source['source_id'],'development')
        transcript=root/DATA/'public-tests'/f't{number:02d}.json'
        row=proof['rows'][number-1]
        if row['task_number']!=number or sha256(transcript)!=row['transcript_sha256']:
            raise ValueError('Different first tool evidence')
        prompt=prefix_prompt(root,common,task,read(transcript))
        if fingerprint(prompt['description'])!=row['description_sha256']:
            raise ValueError('Prefix differs from frozen preflight')
        prompts[number]=prompt
    order=schedule(list(tasks),plan)
    if fingerprint(order)!=plan['schedule_sha256']:
        raise ValueError('Evaluation schedule changed')
    return Context(plan,path,common,tasks,sources,prompts,proof,order,
                   read(root/'private/parent-runtime.json'),files,recovery)


def run(root, context):
    out=root/(RESUMED if context.recovery else OUT)
    if out.exists():
        raise ValueError('Preserve the original registered attempt')
    if len(str(out/'backend.sock').encode())>=108:
        raise ValueError('Backend socket path too long')
    plan, common=context.plan,context.common
    profile=read(root/common['evaluation_profile_path'])
    verify_model(Path(context.runtime['model']),root/common['model_manifest_path'],profile)
    budget=plan['evaluation_budget']
    sandboxes={path:Sandbox(read(root/path),budget['max_tool_seconds'])
               for path in {test_runtime(t['family']) for t in context.tasks.values()}}
    retained=inspect(root/OUT,context) if context.recovery else None
    resume_start=time.time()
    start=retained['started_epoch'] if retained else resume_start
    deadline=start+plan['pipeline_limit_seconds']
    if resume_start>=deadline:
        raise TimeoutError('Original registered deadline has expired')
    if retained:
        restore(root/OUT,out,retained)
    else:
        (out/'private').mkdir(mode=0o700,parents=True)
    state={'stage':'loading','complete':False,'started_epoch':start,'deadline_epoch':deadline,
           'protocol_sha256':sha256(context.plan_path),'schedule_sha256':plan['schedule_sha256'],
           'expected_runs':len(context.order),'runs':[],'backend_lifecycles':[],
           'optimizer_updates':0,'final_test_opened':False,'weights_public':False}
    if retained:
        state.update(runs=retained['runs'],backend_lifecycles=retained['backend_lifecycles'],
                     recovery={'plan_sha256':sha256(root/AMENDMENT),'resumed_epoch':resume_start,
                               'previous_finished_epoch':retained['finished_epoch'],
                               'retained_rollouts':len(retained['runs']),
                               'retained_telemetry':retained['telemetry']})
    remaining=context.order[len(state['runs']):]

    def save():
        atomic(out/'controller.json',state)

    save()
    backend=JournalBackend(root,out,context.runtime,common,state['backend_lifecycles'],save)
    backend.number=len(state['backend_lifecycles'])
    telemetry=Telemetry(out/'telemetry.csv',2 if retained else 1)
    telemetry.start()
    try:
        previous=None
        for item in remaining:
            arm=item['arm']
            planned=budget['max_task_seconds']+budget['max_tool_seconds']+plan['recovery_accounting_seconds']
            if time.time()+planned+(600 if arm!=previous else 0)>deadline:
                raise TimeoutError('Registered pipeline has insufficient budget for another task')
            state.update(stage='evaluating',current_run=item)
            save()
            protocol=effective(common,arm)
            backend.protocol=protocol
            role='base' if arm=='base' else 'student'
            socket=backend.ensure(role,deadline)
            previous=arm
            task=context.prompts[item['task_number']]
            directory=out/'private/tasks'/item['run_id']
            sandbox=sandboxes[test_runtime(task['family'])]
            result=rollout(root,root/DATA,task,context.sources[task['source_id']],socket,sandbox,
                         protocol,directory,role+'_current',journal_path=backend.journal_path,
                         recovery_seconds=plan['recovery_accounting_seconds'])
            backend.check_exclusive()
            result.update(run_id=item['run_id'],contrast_arm=arm,repetition=item['repetition'],
                          task_number=item['task_number'],block=item['block'],backend_index=backend.number)
            write_json(directory/'result.json',result)
            state['runs'].append(result)
            save()
            print(json.dumps({'rollouts_completed':len(state['runs']),'rollouts_expected':len(context.order)}),flush=True)
        state.update(stage='completed',complete=True)
    except BaseException as err:
        state.update(stage='failed',error_type=type(err).__name__)
        (out/'private/error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        backend.stop()
        unchanged=frozen(root,context.files)
        samples=telemetry.stop()
        if retained:
            state['recovery']['new_telemetry']=samples.copy()
            for key in ('samples','errors'):
                samples[key]+=retained['telemetry'][key]
        state.update(frozen_inputs_unchanged=unchanged,backend_stopped=True,telemetry=samples,
                     finished_epoch=time.time(),finished_utc=datetime.now(timezone.utc).isoformat())
        if not unchanged:
            state.update(stage='inputs_changed',complete=False)
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--check',action='store_true')
    p.add_argument('--resume',action='store_true',help='Use the published recovery amendment')
    a=p.parse_args()
    root=a.root.resolve(); context=inputs(root,resume=a.resume)
    if a.check:
        retained=inspect(root/OUT,context) if a.resume else None
        print(json.dumps({'tasks':len(context.tasks),'rollouts':len(context.order),
                          'retained_rollouts':len(retained['runs']) if retained else 0,'final_test_opened':False}))
    else:
        run(root,context)
