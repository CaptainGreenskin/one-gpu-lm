"""Replay the base model, hashed public input, request journal, and grading receipts."""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_tasks import prepare
from one_gpu_lm.verified.rollout import normalization

from evidence import ADMISSION, ARMS, OUTPUT, read
from run import inputs, tasks_with_evidence


def analyze(rows, stuck_numbers, gate):
    grouped = defaultdict(dict)
    for row in rows:
        if not row['audit_passed'] or row['arm'] in grouped[row['task_number']]:
            raise ValueError('Expected one audited result per arm and task')
        grouped[row['task_number']][row['arm']] = row
    if set(grouped) != set(range(1, len(grouped)+1)) or any(set(row)!=set(ARMS) for row in grouped.values()):
        raise ValueError('Expected complete task pairs')
    if not set(stuck_numbers) <= grouped.keys():
        raise ValueError('Missing historical no-edit task')
    stats = Counter()
    by_family = Counter()
    for n, pair in grouped.items():
        a,b = (pair[arm] for arm in ARMS)
        if a['family']!=b['family'] or a['multifile']!=b['multifile']:
            raise ValueError('Mismatched pair')
        delta = int(b['completed_repair'])-int(a['completed_repair'])
        by_family[a['family']] += delta
        stats['new_repairs'] += delta>0
        stats['lost_repairs'] += delta<0
        stats['new_first_edits'] += a['no_edit'] and not b['no_edit']
        stats['lost_first_edits'] += b['no_edit'] and not a['no_edit']
        if n in stuck_numbers:
            stats['historically_stuck_edited_exit_only'] += not a['no_edit']
            stats['historically_stuck_edited_transcript'] += not b['no_edit']
            stats['historically_stuck_repaired_exit_only'] += a['completed_repair']
            stats['historically_stuck_repaired_transcript'] += b['completed_repair']
    net = stats['new_repairs']-stats['lost_repairs']
    return {
        'paired_tasks':len(grouped),'historically_stuck_tasks':len(stuck_numbers),
        **stats,'net_repairs':net,'net_by_family':dict(sorted(by_family.items())),
        'followup_gate_passed':net>=gate['minimum_net_repairs']
            and sum(d>0 for d in by_family.values())>=gate['minimum_positive_families']
            and stats['lost_repairs']<=gate['maximum_regressions'],
    }


def run(root):
    _,ctx,preflight = inputs(root)
    out = root / OUTPUT
    state_path = out / 'controller.json'
    controller = read(state_path)
    if (not controller['complete'] or not controller['backend_stopped'] or not controller['frozen_inputs_unchanged']
            or controller['registration_sha256']!=sha256(ctx.plan_path)
            or controller['schedule_sha256']!=ctx.plan['schedule_sha256']
            or controller['final_test_opened'] or controller['optimizer_updates']!=0
            or [row['run_id'] for row in controller['runs']] != [row['run_id'] for row in ctx.order]):
        raise ValueError('Expected complete registered sweep')
    backends = controller['backend_lifecycles']
    if len(backends)!=1 or backends[0]['role']!='base' or backends[0]['index']!=1:
        raise ValueError('Expected both arms using one load of the pinned base')
    backend = backends[0]
    loc = out/'private/backends/0001-base'
    health = read(loc/'served-artifact.json')
    if (health['adapter'] is not False or health['adapter_sha256']
            or health['profile_sha256']!=ctx.protocol['evaluation_profile_sha256']
            or read(loc.with_name('0001-base-environment.json'))!=read(root/ctx.protocol['environment_path'])):
        raise ValueError('Unexpected model or environment')

    spec = importlib.util.spec_from_file_location('prior035_audit', root/'experiments/035-edit-bottleneck/audit.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ctx.runtime['model'],local_files_only=True,trust_remote_code=False)
    journal = RequestJournal(loc/'requests')
    profile = read(root/ctx.protocol['evaluation_profile_path'])
    norm = normalization(root)
    reconstruction = out/'private/audit'
    reconstruction.mkdir()
    ready = {row['task_number']:row for row in preflight['rows']}
    admission = {row['id']:row for row in read(root/ADMISSION)['rows']}
    workspaces = {}
    tasks = tasks_with_evidence(root,ctx)
    rows = []
    opening = defaultdict(dict)
    seen = set()
    for item,result in zip(ctx.order,controller['runs']):
        run_id = item['run_id']
        task = tasks[run_id]
        number,arm = item['task_number'],item['arm']
        if (result['backend_index']!=1 or result['id']!=task['id']
                or result['task_number']!=number or result['contrast_arm']!=arm):
            raise ValueError('Misbound run')
        source = ctx.sources[task['source_id']]
        if number not in workspaces:
            snapshot = ctx.data/'snapshots'/task['source_id']
            workspaces[number] = prepare(snapshot,reconstruction/'broken'/f't{number:02d}',source,task,hidden=False)
        measure = ready[number]
        broken = workspaces[number]
        if (measure['workspace_sha256']!=fingerprint(sorted(broken.items()))
                or measure['arm_scope']!='shared'):
            raise ValueError('Model run used a different starting workspace than public preflight')
        trial = out/'private/tasks'/run_id
        row,first_input,requests = previous.audited_row(
            root,ctx.data,trial,result,task,source,ctx.protocol,profile,tokenizer,norm,
            journal,reconstruction/'graded'/run_id,broken,backend)
        if seen & requests:
            raise ValueError('Duplicate physical generation request')
        seen.update(requests)
        opening[number][arm] = first_input
        row.update(multifile=admission[task['id']]['multifile'],
                   public_transcript_chars=measure['transcript_chars'] if arm=='transcript' else 0,
                   preflight_failure_marker=measure['visible_failure_marker'],
                   preflight_clipped=measure['normalization']['raw_prefix_clipped'])
        rows.append(row)
    if any(len(inputs)!=2 or len(set(inputs.values()))!=2 for inputs in opening.values()):
        raise ValueError('The evidence change must be visible to every transcript-arm opening')
    if {p.name for p in journal.directory.iterdir() if p.is_dir()} != seen:
        raise ValueError('Unaccounted generation request on the GPU')
    historic = preflight['diagnosis']
    analysis = analyze(rows,set(historic['stuck_tasks']),ctx.plan['followup_gate'])
    value = {
        'schema_version':1,'passed':True,'optimizer_updates':0,'final_test_opened':False,
        'registration_sha256':controller['registration_sha256'],'controller_sha256':sha256(state_path),
        'opening_token_change_verified_pairs':len(opening),'verified_requests':len(seen),
        'public_regressions':preflight['public_regressions'],'public_tool_seconds':preflight['total_tool_seconds'],
        'received_failure_marker':preflight['received_failure_marker'],'clipped_outputs':preflight['clipped_outputs'],
        'base_diagnosis_before':{k:v for k,v in historic.items() if k!='stuck_tasks'},
        'started_epoch':controller['started_epoch'],'finished_epoch':controller['finished_epoch'],
        'telemetry':controller['telemetry'],'arms':previous.totals(rows),'analysis':analysis,'rows':rows,
    }
    write_json(out/'audit.json',value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.root.resolve())
    print(json.dumps({k:result[k] for k in ('passed','verified_requests','opening_token_change_verified_pairs')}))
