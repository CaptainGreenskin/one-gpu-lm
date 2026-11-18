"""Run the registered paired development contrast with durable generation receipts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.verified.first_action_design import ARMS, schedule
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_tasks import adapt, validate_split
from one_gpu_lm.verified.resilient_backend_control import JournalBackend
from one_gpu_lm.verified.resilient_rollout import run as rollout
from one_gpu_lm.verified.rollout import atomic
from one_gpu_lm.verified.sandbox import Sandbox
from one_gpu_lm.verified.tasks import relative, validate_source


EXP = Path('experiments/034-first-action-ablation')
PRIVATE = Path('private/first-action-ablation')
OUTPUT = Path('runs/historic034-development-001')
RECOVERY = 180


def read(path):
    return json.loads(path.read_text())


def frozen(root, files):
    return all((root / relative(name)).is_file() and sha256(root / relative(name)) == h for name,h in files.items())


def inputs(root):
    path=root/EXP/'evaluation-plan.json'
    plan=read(path)
    receipt=read(root/PRIVATE/'eval-publication.json')
    if (not receipt['anonymous_files_verified'] or receipt['ci_conclusion']!='success'
            or receipt['plan_sha256']!=sha256(path) or not frozen(root,plan['frozen_files'])):
        raise ValueError('Development preregistration, CI or frozen inputs differ')
    parent=read(root/relative(plan['032_evaluation_protocol']))
    same=('evaluation_budget','agent_policy','normalization')
    if any(plan[k]!=parent[k] for k in same):
        raise ValueError('Inference setting differs from 032')
    data=root/PRIVATE/'development'
    tasks=read(data/'tasks.json')['tasks'];sources=read(data/'sources.json')['sources']
    admissions=read(data/'admission.json');stable=read(data/'stability.json')
    if (not admissions['complete'] or not stable['passed'] or admissions['model_calls']!=0 or stable['model_calls']!=0
            or admissions['final_test_opened'] or stable['final_test_opened']):
        raise ValueError('CPU-only admission and repeated verification required')
    ids={t['id'] for t in tasks}
    if ids != {r['id'] for r in admissions['rows'] if r['admitted']} or ids != {r['id'] for r in stable['rows'] if r['selected']}:
        raise ValueError('Task set differs from score-blind CPU selection')
    if set(plan['families']) != {t['family'] for t in tasks} or len({s['language'] for s in sources})!=2:
        raise ValueError('Registered family and language coverage required')
    old_sources=read(root/relative(parent['data_directory'])/'sources.json')['sources']
    old_sources+=read(root/relative(parent['training_sources_path']))['sources']
    previous={x for s in old_sources for x in [s['family'],s['repository'],*s.get('related_families',[])]}
    validate_split(sources,previous)
    by_source={s['source_id']:s for s in sources}
    for t in tasks:
        s,_=adapt(by_source[t['source_id']],t)
        validate_source(s,data/'snapshots'/s['source_id'],'development')
    order=schedule(tasks)
    if len(order)!=plan['rollouts'] or fingerprint(order)!=plan['schedule_sha256']:
        raise ValueError('Registered run order differs')
    return plan,parent,data,tasks,by_source,order


def effective(root,parent,plan,training,arm):
    if arm not in ARMS:
        raise ValueError('Unknown training arm')
    result_path=root/'runs'/f'historic034-{arm}-001'/'result.json'
    proof=training['arms'][arm]
    if (sha256(result_path)!=proof['result_sha256'] or read(result_path)['complete'] is not True
            or sha256(result_path.with_name('integrity.json'))!=proof['integrity_sha256']):
        raise ValueError('Trained artifact differs from audited receipt')
    weights=proof['weights_sha256']
    return {**{k:parent[k] for k in ['environment_path','evaluation_profile_path','evaluation_profile_sha256',
                'model_manifest_path','model_id','revision']},
            **{k:plan[k] for k in ['evaluation_budget','agent_policy','normalization']},
            'adapter_directory':str(result_path.parent.relative_to(root)/'private'/'adapter'),
            'adapter_sha256':{k:v for k,v in weights.items() if k.endswith('.safetensors')},
            'all_adapter_files_sha256':weights,'evaluation_variant':arm,'final_test_opened':False}


def run(root):
    root=root.resolve()
    out=root/OUTPUT
    if out.exists():
        raise ValueError('Retain the first attempt and its receipts')
    plan,parent,data,tasks,sources,order=inputs(root)
    training=read(root/PRIVATE/'training-integrity.json')
    if not training['passed'] or training['final_test_opened']:
        raise ValueError('The complete training pair must pass audit')
    runtime=read(root/'private/parent-runtime.json')
    profile=read(root/relative(parent['evaluation_profile_path']))
    verify_model(Path(runtime['model']),root/relative(parent['model_manifest_path']),profile)
    protocol={a:effective(root,parent,plan,training,a) for a in ARMS}
    runtime_map=read(root/PRIVATE/'runtimes.json')
    if len(set(runtime_map.values())) != 1 or set(runtime_map) != set(plan['families']):
        raise ValueError('All families must use the registered offline runtime')
    runtime_path=Path(next(iter(runtime_map.values())))
    if runtime_path.resolve()!=(root/PRIVATE/'offline-runtime.json').resolve():
        raise ValueError('Runtime location differs from the frozen input')
    runner=Sandbox(read(runtime_path),plan['evaluation_budget']['max_tool_seconds'])
    by_id={t['id']:t for t in tasks}
    out.mkdir(mode=0o700)
    (out/'private').mkdir(mode=0o700)
    started=time.time()
    limit=started+plan['pipeline_limit_seconds']
    state={'stage':'starting','complete':False,'started_epoch':started,'optimizer_updates':0,
           'training_integrity_sha256':sha256(root/PRIVATE/'training-integrity.json'),
           'registration_sha256':sha256(root/EXP/'evaluation-plan.json'),'schedule_sha256':plan['schedule_sha256'],
           'runs':[],'backend_lifecycles':[],'training_export_allowed':False,'final_test_opened':False,'weights_public':False}
    def save():
        atomic(out/'controller.json',state)
    backend=JournalBackend(root,out,runtime,protocol[ARMS[0]],state['backend_lifecycles'],save)
    telemetry=Telemetry(out/'telemetry.csv',1)
    telemetry.start();save()
    last_arm=None
    try:
        for row in order:
            arm=row['arm']
            if time.time()+plan['evaluation_budget']['max_task_seconds']+RECOVERY+660>limit:
                raise TimeoutError('Campaign wall clock limit reached')
            state.update(stage='loading' if arm!=last_arm else 'evaluating', current={k:v for k,v in row.items() if k!='task_id'})
            save()
            if arm!=last_arm:
                backend.stop()
                backend.protocol=protocol[arm]
            socket=backend.ensure('student',limit)
            backend.entry['arm']=arm
            last_arm=arm
            state['stage']='evaluating';save()
            task=by_id[row['task_id']]
            directory=out/'private/tasks'/row['run_id']
            result=rollout(root,data,task,sources[task['source_id']],socket,runner,protocol[arm],directory,
                           'student_current',journal_path=backend.journal_path,recovery_seconds=RECOVERY)
            backend.check_exclusive()
            result.update(contrast_arm=arm,run_id=row['run_id'],backend_index=backend.number,block=row['block'],task_number=row['task_number'])
            write_json(directory/'result.json',result)
            state['runs'].append(result);save()
            print(json.dumps({'complete_rollouts':len(state['runs']),'planned_rollouts':len(order),'arm':arm}),flush=True)
        state.update(stage='completed',complete=True)
    except BaseException as error:
        state.update(stage='failed',error_type=type(error).__name__)
        (out/'private/error.log').write_text(traceback.format_exc())
        raise
    finally:
        backend.stop()
        state.update(telemetry=telemetry.stop(),backend_stopped=True,finished_epoch=time.time(),
                     finished_utc=datetime.now(timezone.utc).isoformat(),frozen_inputs_unchanged=frozen(root,plan['frozen_files']))
        if not state['frozen_inputs_unchanged']:
            state.update(stage='input_changed',complete=False)
        save()
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    run(args.root)
