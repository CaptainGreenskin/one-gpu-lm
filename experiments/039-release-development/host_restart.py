"""Retain the audited 039 prefix across the workstation restart."""
import json
from pathlib import Path
import shutil
import tempfile

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.serving.receipt_client import payload_sha256, request_identity
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.resilient_eval_audit import audit_task
from one_gpu_lm.verified.resume_prefix import clean_samples

from recovery import AMENDMENT, EXP, RESUMED, read


RESTART_AMENDMENT = EXP/'host-restart-plan.json'
RESTARTED = Path('runs/q039-development-restart')
RESTART_STATUS = 'pipeline-restarted.json'
RESTART_LAUNCH = 'development-restart-launch.json'
PRIVATE = Path('private/modelscope-development')


def validate_amendment(root, plan_path, plan, files):
    path = root/RESTART_AMENDMENT
    amendment = read(path)
    receipt = read(root/PRIVATE/'restart-publication.json')
    changed, added = amendment['frozen_replacements'], amendment['frozen_additions']
    if (amendment['schema_version'] != 1 or amendment['original_protocol_sha256'] != sha256(plan_path)
            or amendment['first_recovery_plan_sha256'] != sha256(root/AMENDMENT)
            or amendment['schedule_sha256'] != plan['schedule_sha256']
            or set(changed) != {str(EXP/'run.py'),str(EXP/'audit.py')}
            or set(added) != {str(EXP/'host_restart.py')}
            or receipt['restart_plan_sha256'] != sha256(path) or receipt['protocol_sha256'] != sha256(plan_path)
            or receipt['anonymous_files_verified'] is not True or receipt['ci_conclusion'] != 'success'):
        raise ValueError('Require the verified public restart amendment')
    return amendment,{**files,**changed,**added,str(RESTART_AMENDMENT):sha256(path)}


def interrupted(root, context, original):
    """Verify the preserved prefix and account for any unfinished actor requests."""
    prior = root/RESUMED
    plan = context.restart['incident']
    paths = {'prior_controller_sha256':prior/'controller.json','prior_telemetry_sha256':prior/'telemetry.csv',
             'prior_pipeline_sha256':root/PRIVATE/'pipeline-resumed.json',
             'prior_launch_sha256':root/PRIVATE/'development-resume-launch.json'}
    if any(sha256(path) != plan[name] for name,path in paths.items()):
        raise ValueError('Interrupted attempt differs from registered evidence')
    state = read(paths['prior_controller_sha256'])
    runs, backends = state['runs'], state['backend_lifecycles']
    split = plan['retained_rollouts']
    first = state['recovery']
    boot = plan['boot_epoch']
    if (state['stage'] != 'evaluating' or state['complete'] or state['final_test_opened'] or state['weights_public']
            or state['optimizer_updates'] or state['protocol_sha256'] != sha256(context.plan_path)
            or state['schedule_sha256'] != context.plan['schedule_sha256']
            or state['started_epoch'] != original['started_epoch'] or state['deadline_epoch'] != original['deadline_epoch']
            or not state['started_epoch'] < boot < state['deadline_epoch']
            or first['plan_sha256'] != sha256(root/AMENDMENT) or first['retained_telemetry'] != original['telemetry']
            or first['retained_rollouts'] != len(original['runs']) or runs[:len(original['runs'])] != original['runs']
            or backends[:len(original['backend_lifecycles'])] != original['backend_lifecycles']
            or len(runs) != split or not len(original['runs']) < split < len(context.order)
            or [r['run_id'] for r in runs] != [r['run_id'] for r in context.order[:split]]
            or state['current_run'] != context.order[split] or len(context.order)-split != plan['remaining_rollouts']
            or fingerprint(context.order[split:]) != plan['remaining_schedule_sha256']):
        raise ValueError('Restart differs from the original schedule, prefix or deadline')

    folders = prior/'private/backends'
    journals, served, retained = {},{},{}
    for index, entry in enumerate(backends,1):
        role = entry['role']
        directory = folders/f'{index:04d}-{role}'
        artifact = read(directory/'served-artifact.json')
        if (entry['index'] != index or role not in {'base','student'}
                or not entry['started_epoch'] <= entry['ready_epoch'] <= entry.get('stopped_epoch',boot) <= boot
                or (index == len(backends)) == ('stopped_epoch' in entry)
                or artifact['model_id'] != context.common['model_id'] or artifact['revision'] != context.common['revision']
                or artifact['profile_sha256'] != context.common['evaluation_profile_sha256']
                or artifact['request_journal_version'] != 1 or artifact['ready'] is not True
                or artifact['adapter'] is not (role == 'student')
                or artifact['adapter_sha256'] != (context.common['adapter_sha256'] if role == 'student' else {})
                or not (directory/'requests').is_dir()):
            raise ValueError('Archived model-serving identity or time bounds differ')
        journals[index],served[index],retained[index] = RequestJournal(directory/'requests'),artifact,set()

    tasks = prior/'private/tasks'
    for result in runs:
        directory = tasks/result['run_id']
        trace_path = directory/'trajectory.json'
        trace = read(trace_path)
        index = result['backend_index']
        entry = backends[index-1]
        if (read(directory/'result.json') != result or sha256(trace_path) != result['trajectory_sha256']
                or not trace['complete'] or trace['backend'] != served[index]
                or not entry['ready_epoch'] <= trace['started_epoch'] <= trace['finished_epoch'] <= entry.get('stopped_epoch',boot)):
            raise ValueError('A completed result or trajectory changed after interruption')
        requests = audit_task(directory,trace,result,journals[index])
        if len(requests) != result['attempted_calls'] or requests & retained[index]:
            raise ValueError('Archived complete requests must reconcile once per task')
        retained[index].update(requests)

    job = state['current_run']['run_id']
    folder = tasks/job
    partial_path = folder/'trajectory.json'
    partial = read(partial_path)
    index = len(backends)
    cut_ids,receipts = set(),[]
    if (partial['complete'] or (folder/'result.json').exists() or partial['backend'] != served[index]
            or not 1 <= partial['attempted_calls'] <= context.plan['evaluation_budget']['max_model_calls']
            or sha256(partial_path) != plan['interrupted_trajectory_sha256']):
        raise ValueError('The interrupted task differs from its frozen receipt')
    for call in range(1,partial['attempted_calls']+1):
        identity = request_identity(job,call)
        receipt = journals[index].status(identity)
        if receipt['state'] == 'not_received':
            continue
        payload = json.loads((folder/f'request-{call:03d}.json').read_text())
        if (receipt['request_id'] != identity or receipt['payload_sha256'] != payload_sha256(payload)
                or not partial['started_epoch'] <= receipt['accepted_epoch'] <= receipt.get('finished_epoch',boot) <= boot):
            raise ValueError('The interrupted backend receipt differs from the actor request')
        cut_ids.add(identity)
        receipts.append(receipt)
    complete = [r['result'] for r in receipts if r['state'] == 'completed']
    telemetry, damaged, last = clean_samples(prior/'telemetry.csv')
    observed = {'valid_telemetry_samples':len(telemetry.splitlines())-1,'damaged_telemetry_rows':damaged,
                'last_pre_restart_sample_epoch':last,'prior_backend_count':len(backends),
                'interrupted_attempted_calls':partial['attempted_calls'],'interrupted_journaled_requests':len(receipts),
                'interrupted_complete_responses':len(complete),
                'interrupted_unknown_responses':sum(r['unknown_generation_cost'] for r in receipts),
                'known_interrupted_input_tokens':sum(r['input_tokens'] for r in complete),
                'known_interrupted_generated_tokens':sum(r['generated_tokens'] for r in complete)}
    if (any(plan[k] != v for k,v in observed.items()) or last >= boot or cut_ids & retained[index]
            or [s['generation'] for s in partial['steps']] != complete[:len(partial['steps'])]
            or any({p.name for p in journal.directory.iterdir()} != retained[i] | (cut_ids if i == index else set())
                   for i,journal in journals.items())):
        raise ValueError('Archived request or telemetry accounting differs from registration')
    backends[-1].update(stopped_epoch=boot,stop_timestamp_is_upper_bound=True)
    return state,telemetry,cut_ids


def restore(root, destination, state, telemetry, cut_ids):
    """Copy completed artifacts into a new attempt and retain the interrupted appendices in place."""
    origin = root/RESUMED/'private'
    last = state['backend_lifecycles'][-1]
    interrupted_journal = origin/'backends'/f"{last['index']:04d}-{last['role']}"/'requests'

    def ignore(folder, names):
        return cut_ids.intersection(names) if Path(folder) == interrupted_journal else set()

    with tempfile.TemporaryDirectory(prefix='039-restore-',dir=destination.parent) as temporary:
        stage = Path(temporary)/'attempt'
        tasks = stage/'private/tasks'
        tasks.mkdir(mode=0o700,parents=True)
        for record in state['runs']:
            jid = record['run_id']
            shutil.copytree(origin/'tasks'/jid,tasks/jid,symlinks=True)
        shutil.copytree(origin/'backends',stage/'private/backends',ignore=ignore)
        (stage/'telemetry.csv').write_bytes(telemetry)
        if destination.exists():
            raise FileExistsError('Preserve the existing restart attempt')
        stage.rename(destination)
