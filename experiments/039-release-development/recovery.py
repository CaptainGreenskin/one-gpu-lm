"""Validate the first 039 attempt and retain only its complete scheduled prefix."""
import json
from pathlib import Path
import shutil

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.resilient_eval_audit import audit_task
from one_gpu_lm.verified.resume_prefix import clean_samples


EXP = Path('experiments/039-release-development')
AMENDMENT = EXP/'recovery-plan.json'
RESUMED = Path('runs/q039-development-resume')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def served_weight_hashes(files):
    """The journaled backend serves tensor digests and verifies all files before load."""
    expected = {name:value for name,value in files.items() if name.endswith('.safetensors')}
    if set(expected)!={'adapter_model.safetensors'} or 'adapter_config.json' not in files:
        raise ValueError('Expected the exact registered adapter bundle')
    return expected


def validate_amendment(root, plan_path, plan, publication_path):
    path = root/AMENDMENT
    amendment = read(path)
    publication = read(publication_path)
    changed = amendment['frozen_replacements']
    added = amendment['frozen_additions']
    if (amendment['original_protocol_sha256']!=sha256(plan_path)
            or amendment['schedule_sha256']!=plan['schedule_sha256']
            or set(changed)!={str(EXP/'run.py'),str(EXP/'audit.py')}
            or set(added)!={str(EXP/'recovery.py')}
            or publication['recovery_plan_sha256']!=sha256(path)
            or publication['protocol_sha256']!=sha256(plan_path)
            or publication['anonymous_files_verified'] is not True or publication['ci_conclusion']!='success'):
        raise ValueError('Expected the verified public recovery amendment')
    return amendment, {**plan['frozen_files'], **changed, **added, str(AMENDMENT):sha256(path)}


def inspect(original, context):
    """Reconcile every original base request and prove the failed candidate started none."""
    proof = context.recovery['incident']
    state = read(original/'controller.json')
    runs, backends = state['runs'], state['backend_lifecycles']
    split = proof['retained_rollouts']
    if (sha256(original/'controller.json')!=proof['original_controller_sha256']
            or sha256(original/'telemetry.csv')!=proof['original_telemetry_sha256']
            or state['complete'] or state['stage']!='failed' or state['error_type']!='ValueError'
            or not state['backend_stopped'] or not state['frozen_inputs_unchanged']
            or state['final_test_opened'] or state['weights_public'] or state['optimizer_updates']
            or state['protocol_sha256']!=sha256(context.plan_path)
            or state['schedule_sha256']!=context.plan['schedule_sha256']
            or len(runs)!=split or split!=context.plan['block_size']
            or [r['run_id'] for r in runs]!=[item['run_id'] for item in context.order[:split]]
            or state['current_run']!=context.order[split] or state['current_run']['arm']!='candidate'
            or state['deadline_epoch']!=state['started_epoch']+context.plan['pipeline_limit_seconds']
            or not state['started_epoch']<state['finished_epoch']<state['deadline_epoch']
            or not state['telemetry']['thread_stopped']):
        raise ValueError('The original attempt differs from the registered incident')
    if [(entry['index'],entry['role']) for entry in backends]!=[(1,'base'),(2,'student')]:
        raise ValueError('Expected one complete base load and one interrupted candidate load')
    parent = original/'private/backends'
    served = {}
    for entry in backends:
        i,role = entry['index'],entry['role']
        directory = parent/f'{i:04d}-{role}'
        artifact = read(directory/'served-artifact.json')
        if (not entry['started_epoch']<=entry['ready_epoch']<=entry['stopped_epoch']<=state['finished_epoch']
                or artifact['model_id']!=context.common['model_id'] or artifact['revision']!=context.common['revision']
                or artifact['profile_sha256']!=context.common['evaluation_profile_sha256']
                or artifact['request_journal_version']!=1 or artifact['ready'] is not True
                or artifact['adapter'] is not (i==2)
                or artifact['adapter_sha256']!=(context.common['adapter_sha256'] if i==2 else {})):
            raise ValueError('The original backend did not serve the registered identity')
        served[i] = artifact
    tasks = original/'private/tasks'
    aborted = tasks/state['current_run']['run_id']
    failed_requests = parent/'0002-student/requests'
    if (proof['failed_backend_index']!=2 or proof['aborted_generation_requests']!=0
            or {p.name for p in tasks.iterdir()}!={aborted.name, *(run['run_id'] for run in runs)}
            or {p.name for p in aborted.iterdir()}!={'workspace'}
            or not failed_requests.is_dir() or any(failed_requests.iterdir())):
        raise ValueError('The failed candidate may already have generated or edited')
    journal_dir = parent/'0001-base/requests'
    if not journal_dir.is_dir():
        raise ValueError('Missing original base request journal')
    journal = RequestJournal(journal_dir)
    seen = set()
    for item, result in zip(context.order,runs):
        run = tasks/result['run_id']
        trace = read(run/'trajectory.json')
        if (result['backend_index']!=1 or result['contrast_arm']!='base' or item['arm']!='base'
                or not trace['complete'] or trace['id']!=context.tasks[item['task_number']]['id']
                or trace['backend']!=served[1] or read(run/'result.json')!=result
                or sha256(run/'trajectory.json')!=result['trajectory_sha256']
                or not backends[0]['ready_epoch']<=trace['started_epoch']<=trace['finished_epoch']<=backends[0]['stopped_epoch']):
            raise ValueError('A retained base trajectory differs from its original receipt')
        requests = audit_task(run,trace,result,journal)
        if seen & requests or len(requests)!=result['attempted_calls']:
            raise ValueError('Original request was missing or counted twice')
        seen.update(requests)
    if seen!={p.name for p in journal_dir.iterdir()} or len(seen)!=proof['retained_verified_requests']:
        raise ValueError('The base request journal contains unaccounted requests')
    samples, damaged, last = clean_samples(original/'telemetry.csv')
    if damaged or len(samples.splitlines())!=state['telemetry']['samples']+1 or not state['started_epoch']<=last<=state['finished_epoch']:
        raise ValueError('The original telemetry is incomplete')
    if fingerprint(context.order[split:])!=context.recovery['remaining_schedule_sha256']:
        raise ValueError('The continuation schedule differs from the preregistration')
    return state


def restore(original, destination, state):
    """Preserve the original attempt and copy complete evidence to the continuation."""
    tasks = destination/'private/tasks'
    tasks.mkdir(mode=0o700, parents=True)
    origin = original/'private'
    for result in state['runs']:
        name = result['run_id']
        shutil.copytree(origin/'tasks'/name, tasks/name, symlinks=True)
    shutil.copytree(origin/'backends', destination/'private/backends')
    shutil.copy2(original/'telemetry.csv',destination/'telemetry.csv')
