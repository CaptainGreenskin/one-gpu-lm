"""Audit the two original 036 segments and account for the discarded actor separately."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_tasks import prepare
from one_gpu_lm.verified.rollout import normalization

from audit_results import analyze
from evidence import ADMISSION, ARMS, OUTPUT, read
from recover import RECOVERED, RECONSTRUCTION, RECOVERY_PLAN, inventory, recovery_inputs, retired_segment
from run import tasks_with_evidence


def audit_identity(ctx, location, backend):
    if backend['index']!=1 or backend['role']!='base':
        raise ValueError('Wrong backend identity for physical segment')
    health = read(location/'served-artifact.json')
    if (health['adapter'] is not False or health['adapter_sha256']
            or health['profile_sha256']!=ctx.protocol['evaluation_profile_sha256']):
        raise ValueError('Wrong pinned base on a recovery segment')
    return health


def run(root):
    _,ctx,preflight,recovery = recovery_inputs(root)
    prefix,original_backend,expected_prefix,interrupted = retired_segment(root,ctx,recovery)
    tail_path = root/RECOVERED
    tail = read(tail_path/'controller.json')
    split = recovery['interrupt']['completed_runs']
    active = tail['backend_lifecycles']
    if (not tail['complete'] or not tail['backend_stopped'] or not tail['frozen_inputs_unchanged']
            or tail['optimizer_updates']!=0 or tail['final_test_opened']
            or tail['registration_sha256']!=sha256(root/RECOVERY_PLAN)
            or tail['schedule_sha256']!=recovery['remaining_schedule_sha256']
            or [r['run_id'] for r in tail['runs']]!=[row['run_id'] for row in ctx.order[split:]]
            or not recovery['interrupt']['boot_epoch'] <= tail['started_epoch'] <= tail['finished_epoch'] <=
                   recovery['interrupt']['original_deadline_epoch']
            or len(active)!=1 or not tail['telemetry']['thread_stopped']):
        raise ValueError('The complete frozen continuation must end inside the original deadline')

    spec = importlib.util.spec_from_file_location('prior035_audit', root/'experiments/035-edit-bottleneck/audit.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ctx.runtime['model'],local_files_only=True,trust_remote_code=False)
    profile = read(root/ctx.protocol['evaluation_profile_path'])
    environment = read(root/ctx.protocol['environment_path'])
    norm = normalization(root)
    out = root/RECONSTRUCTION
    out.mkdir(mode=0o700)
    task_map = tasks_with_evidence(root,ctx)
    admission = {row['id']:row for row in read(root/ADMISSION)['rows']}
    ready = {row['task_number']:row for row in preflight['rows']}
    workspaces = {}
    opening = defaultdict(dict)
    rows = []
    counts = []
    segments = (
        (root/OUTPUT,prefix['runs'],original_backend,ctx.order[:split]),
        (tail_path,tail['runs'],active[0],ctx.order[split:]),
    )
    for number,(parent,results,backend,order) in enumerate(segments,1):
        location = parent/'private/backends/0001-base'
        served = audit_identity(ctx,location,backend)
        if read(location.with_name('0001-base-environment.json')) != environment:
            raise ValueError('Backend environment differs across recovery')
        journal = RequestJournal(location/'requests')
        seen = set()
        for item,result in zip(order,results):
            task_number,arm,run_id = (item[k] for k in ('task_number','arm','run_id'))
            task = task_map[run_id]
            if (result['backend_index']!=1 or result['id']!=task['id'] or result['task_number']!=task_number
                    or result['contrast_arm']!=arm):
                raise ValueError('Wrong segment or registered task')
            source = ctx.sources[task['source_id']]
            if task_number not in workspaces:
                snapshot = ctx.data/'snapshots'/task['source_id']
                workspaces[task_number] = prepare(snapshot,out/'broken'/f't{task_number:02d}',source,task,hidden=False)
            before = workspaces[task_number]
            pre = ready[task_number]
            if pre['workspace_sha256']!=fingerprint(sorted(before.items())) or pre['arm_scope']!='shared':
                raise ValueError('Public evidence or score starts from a different workspace')
            trial = parent/'private/tasks'/run_id
            if read(trial/'trajectory.json')['backend']!=served:
                raise ValueError('Trajectory and backend service receipt disagree')
            row,first,requests = previous.audited_row(
                root,ctx.data,trial,result,task,source,ctx.protocol,profile,tokenizer,norm,
                journal,out/'graded'/run_id,before,backend)
            if seen & requests:
                raise ValueError('Duplicate scored generation request')
            seen.update(requests)
            opening[task_number][arm] = first
            row.update(multifile=admission[task['id']]['multifile'],
                       public_transcript_chars=pre['transcript_chars'] if arm=='transcript' else 0,
                       preflight_failure_marker=pre['visible_failure_marker'],
                       preflight_clipped=pre['normalization']['raw_prefix_clipped'])
            rows.append(row)
        if number==1:
            valid = seen==expected_prefix and not (seen & interrupted) and inventory(journal.directory)==seen|interrupted
        else:
            valid = inventory(journal.directory)==seen
        if not valid:
            raise ValueError('Unaccounted physical request in a recovery segment')
        counts.append(len(seen))

    if any(set(pair)!=set(ARMS) or len(set(pair.values()))!=2 for pair in opening.values()):
        raise ValueError('Prompt treatment must appear in every matched opening')
    historic = preflight['diagnosis']
    analysis = analyze(rows,set(historic['stuck_tasks']),ctx.plan['followup_gate'])
    value = {
        'schema_version':1,'passed':True,'optimizer_updates':0,'final_test_opened':False,
        'registration_sha256':sha256(ctx.plan_path),'controller_sha256':sha256(tail_path/'controller.json'),
        'opening_token_change_verified_pairs':len(opening),'verified_requests':sum(counts),
        'public_regressions':preflight['public_regressions'],'public_tool_seconds':preflight['total_tool_seconds'],
        'received_failure_marker':preflight['received_failure_marker'],'clipped_outputs':preflight['clipped_outputs'],
        'base_diagnosis_before':{k:v for k,v in historic.items() if k!='stuck_tasks'},
        'started_epoch':prefix['started_epoch'],'finished_epoch':tail['finished_epoch'],
        'telemetry':tail['telemetry'],'arms':previous.totals(rows),'analysis':analysis,'rows':rows,
        'host_recovery': {
            **recovery['interrupt'],
            'recovery_plan_sha256':sha256(root/RECOVERY_PLAN),
            'resumed_epoch':tail['started_epoch'],'retained_rollouts':split,'resumed_rollouts':len(tail['runs']),
            'retained_verified_requests':counts[0],'resumed_verified_requests':counts[1],
            'physical_backend_loads':2,'score_excludes_interrupted_actor':True,
            'physical_accepted_requests':sum(counts)+len(interrupted),
            'maximum_unobserved_generation_tokens':recovery['maximum_unobserved_generation_tokens'],
            'finished_by_original_deadline':True,
        },
    }
    write_json(tail_path/'audit.json',value)
    return value


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    result = run(args.root.resolve())
    print(json.dumps({k:result[k] for k in ('passed','verified_requests','opening_token_change_verified_pairs')}))
