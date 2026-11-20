"""Resume only the unscored portion of 036 after a verified workstation reboot."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.serving.receipt_client import payload_sha256, request_identity
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.resilient_eval_audit import audit_task
from one_gpu_lm.verified.resume_prefix import clean_samples

from evidence import EXP, OUTPUT, PRIVATE, read
from run import inputs as original_inputs, tasks_with_evidence


RECOVERED = Path('runs/historic036-evidence-r1')
RECONSTRUCTION = Path('runs/historic036-evidence-audit-r1')
RECOVERY_PLAN = EXP / 'recovery-plan.json'


def inventory(path):
    return {entry.name for entry in path.iterdir() if entry.is_dir()}


def retired_segment(root, ctx, recovery):
    """Validate completed rows and account for the unscored interrupted actor."""
    old = root / OUTPUT
    state = read(old/'controller.json')
    proof = recovery['interrupt']
    split = proof['completed_runs']
    deadline = state['started_epoch'] + ctx.plan['pipeline_limit_seconds']
    if (not state['started_epoch'] < proof['boot_epoch'] < deadline or deadline != proof['original_deadline_epoch']
            or state['complete'] or state['optimizer_updates'] or state['final_test_opened']
            or state['registration_sha256']!=sha256(ctx.plan_path)
            or state['schedule_sha256']!=ctx.plan['schedule_sha256']
            or [item['run_id'] for item in state['runs']]!=[r['run_id'] for r in ctx.order[:split]]
            or fingerprint(ctx.order[split:])!=recovery['remaining_schedule_sha256']
            or fingerprint(ctx.order)!=recovery['original_schedule_sha256']
            or not 0 < split < len(ctx.order) or split%2 or state['current_task_number']!=ctx.order[split]['task_number']):
        raise ValueError('Recovery must preserve the exact original complete-pair prefix')
    backends = state['backend_lifecycles']
    if len(backends)!=1 or backends[0]['role']!='base' or backends[0]['index']!=1 or 'stopped_epoch' in backends[0]:
        raise ValueError('Expected one original backend interrupted by host restart')
    backend = dict(backends[0], stopped_epoch=proof['boot_epoch'])
    journal = RequestJournal(old/'private/backends/0001-base/requests')
    seen = set()
    for item in state['runs']:
        path = old/'private/tasks'/item['run_id']
        trace = read(path/'trajectory.json')
        if (item['backend_index']!=1 or not trace['complete']
                or sha256(path/'trajectory.json')!=item['trajectory_sha256'] or read(path/'result.json')!=item
                or not backend['ready_epoch'] <= trace['finished_epoch'] <= backend['stopped_epoch']):
            raise ValueError('Retained trajectory differs')
        requests = audit_task(path,trace,item,journal)
        if seen & requests or len(requests)!=item['attempted_calls']:
            raise ValueError('Duplicate or missing original scored journal request')
        seen.update(requests)
    if len(seen)!=recovery['retained_verified_requests']:
        raise ValueError('Pre-reboot scored generation count differs')

    cut = old/'private/tasks'/ctx.order[split]['run_id']
    trace = read(cut/'trajectory.json')
    budget = ctx.plan['evaluation_budget']
    call_count = trace['attempted_calls']
    if (sha256(cut/'trajectory.json')!=proof['partial_trajectory_sha256'] or trace['complete']
            or trace['boundary']!='in_flight' or trace['split']!='development' or trace['training_export_allowed']
            or trace['id']!=ctx.order[split]['task_id'] or trace['submitted']
            or call_count!=proof['aborted_attempted_calls'] or len(trace['steps'])!=call_count-1
            or (cut/f'tool-{call_count:03d}.json').exists() or (cut/'result.json').exists()):
        raise ValueError('Host interruption did not stop at the registered incomplete action')
    observed = set()
    received = []
    damaged = 0
    upper_generated = 0
    for call in range(1,call_count+1):
        request_id = request_identity(cut.name,call)
        item = journal.status(request_id)
        observed.add(request_id)
        file = cut/f'request-{call:03d}.json'
        if call<call_count:
            step = trace['steps'][call-1]
            payload = read(file)
            if (item['state']!='completed' or item['unknown_generation_cost']
                    or step['call']!=call or item['result']!=step['generation']
                    or sha256(file)!=step['request_sha256']):
                raise ValueError('Prior step in aborted actor is damaged')
            received.append(item['result'])
        else:
            upper_generated = min(budget['max_total_generated_tokens']-trace['generated_tokens'],
                                  budget['max_generated_tokens_per_call'])
            payload = {'operation':'generate','messages':trace['messages'],
                       'agent_policy':ctx.plan['agent_policy'],'max_input_tokens':budget['max_input_tokens'],
                       'max_new_tokens':upper_generated}
            damaged += int(file.stat().st_size==0)
            if (item['state']!='in_flight_or_interrupted' or not item['unknown_generation_cost']
                    or 'result' in item or file.stat().st_size and read(file)!=payload):
                raise ValueError('Interrupted response state differs')
        if (item['request_id']!=request_id or item['payload_sha256']!=payload_sha256(payload)
                or not trace['started_epoch'] <= item['accepted_epoch'] < proof['boot_epoch']):
            raise ValueError('Archived request does not match the accepted journal hash')
    known_generated = sum(v['generated_tokens'] for v in received)
    if (known_generated!=trace['generated_tokens'] or known_generated!=proof['known_generated_tokens']
            or sum(v['input_tokens'] for v in received)!=proof['known_input_tokens']
            or len(received)!=proof['completed_responses'] or call_count-len(received)!=proof['unknown_responses']
            or damaged!=proof['damaged_request_payloads']
            or upper_generated!=recovery['maximum_unobserved_generation_tokens']
            or seen & observed or inventory(journal.directory)!=seen|observed):
        raise ValueError('Interrupted cost inventory differs')
    samples,damaged_rows,last = clean_samples(old/'telemetry.csv')
    if (damaged_rows!=proof['damaged_preboot_samples'] or last!=proof['last_preboot_gpu_sample_epoch']
            or len(samples.splitlines())-1!=proof['archived_preboot_samples']):
        raise ValueError('Archived GPU sampling changed')
    return state,backend,seen,observed


def recovery_inputs(root):
    parent,ctx,preflight = original_inputs(root)
    plan_path = root/RECOVERY_PLAN
    recovery = read(plan_path)
    publication = read(root/PRIVATE/'recovery-publication.json')
    if (not publication['anonymous_files_verified'] or publication['ci_conclusion']!='success'
            or publication['plan_sha256']!=sha256(plan_path)
            or recovery['original_plan_sha256']!=sha256(ctx.plan_path)
            or not ctx.legacy.frozen(root,recovery['frozen_files'])
            or any(recovery[k]!=ctx.plan[k] for k in ('evaluation_budget','agent_policy','normalization',
                                                      'recovery_accounting_seconds','followup_gate'))
            or recovery['final_test_opened'] or recovery['optimizer_updates']!=0):
        raise ValueError('Exact frozen recovery plan and publication required')
    return parent,ctx,preflight,recovery


def budgeted_context(root, ctx, recovery):
    path = root/RECOVERY_PLAN
    # Count downtime against the original absolute cap, leaving a margin for reloading.
    remaining = int(recovery['interrupt']['original_deadline_epoch'] - time.time()) - 300
    if remaining < 1800:
        raise ValueError('Original deadline has insufficient time for recovery')
    frozen = {**ctx.plan['frozen_files'], **recovery['frozen_files'],
              str(RECOVERY_PLAN):sha256(path)}
    return replace(ctx,plan_path=path,plan={**ctx.plan,'pipeline_limit_seconds':remaining,
            'schedule_sha256':recovery['remaining_schedule_sha256'],'frozen_files':frozen},
            order=ctx.order[recovery['interrupt']['completed_runs']:])


def resume(root):
    parent,ctx,_,recovery = recovery_inputs(root)
    retired_segment(root,ctx,recovery)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():
        raise ValueError('GPU is in use')
    if (root/RECOVERED).exists():
        raise ValueError('Preserve first restarted segment')
    tail = budgeted_context(root,ctx,recovery)
    tasks = tasks_with_evidence(root,tail)

    def perform(root,data,task,source,backend,sandbox,protocol,directory,arm,**kwargs):
        presented = tasks[directory.name]
        if presented['id']!=task['id']:
            raise ValueError('Misbinding of recovery task')
        return parent.rollout(root,data,presented,source,backend,sandbox,protocol,directory,arm,**kwargs)

    parent.sweep(root,root/RECOVERED,tail,perform=perform)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    if args.check:
        _,ctx,_,recovery=recovery_inputs(root)
        _,_,seen,interrupted=retired_segment(root,ctx,recovery)
        print(json.dumps({'retained_runs':recovery['interrupt']['completed_runs'],
                          'retained_requests':len(seen),'retained_interrupted_requests':len(interrupted),
                          'remaining_runs':len(budgeted_context(root,ctx,recovery).order)}))
    else:
        resume(root)
