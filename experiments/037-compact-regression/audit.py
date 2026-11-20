"""Match every 037 rollout to the frozen prompt, tool receipt and protected grade."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_tasks import prepare as materialize
from one_gpu_lm.verified.rollout import normalization

from prepare import ARMS, OUT, module, read
from run import inputs, variants


def analyze(rows, gate, n):
    pairs = defaultdict(dict)
    for row in rows:
        arm = row['arm']
        number = row['task_number']
        if not row['audit_passed'] or arm not in ARMS or arm in pairs[number]:
            raise ValueError('Unexpected scored row')
        pairs[number][arm] = row
    if set(pairs)!=set(range(1,n+1)) or any(set(pair)!=set(ARMS) for pair in pairs.values()):
        raise ValueError('Full pairs required')
    stats = Counter()
    families = Counter()
    for pair in pairs.values():
        old,new = (pair[arm] for arm in ARMS)
        if any(old[k]!=new[k] for k in ('family','multifile','old_prefix_clipped','recovered_failure_line','historically_stuck')):
            raise ValueError('Cohort metadata differs')
        outcome = int(new['completed_repair']) - int(old['completed_repair'])
        families[old['family']] += outcome
        stats['improvements'] += outcome>0
        stats['regressions'] += outcome<0
        stats['new_edits'] += old['no_edit'] and not new['no_edit']
        stats['lost_edits'] += new['no_edit'] and not old['no_edit']
        for subset in ('old_prefix_clipped','recovered_failure_line','historically_stuck'):
            if old[subset]:
                stats[subset+'_pairs'] += 1
                for arm, r in zip(ARMS, (old,new)):
                    stats[subset+'_repairs_'+arm] += r['completed_repair']
                    stats[subset+'_edited_'+arm] += not r['no_edit']
    net = stats['improvements']-stats['regressions']
    return {'paired_tasks':len(pairs), **stats, 'net_repairs':net,'net_by_family':dict(sorted(families.items())),
            'followup_gate_passed':net>=gate['minimum_net_repairs']
                and sum(d>0 for d in families.values())>=gate['minimum_positive_families']
                and stats['regressions']<=gate['maximum_regressions']}


def run(root):
    _,ctx,proof,legacy = inputs(root)
    out = root/OUT
    path = out/'controller.json'
    state = read(path)
    if (not state['complete'] or not state['backend_stopped'] or not state['frozen_inputs_unchanged']
            or state['final_test_opened'] or state['optimizer_updates']!=0
            or state['registration_sha256']!=sha256(ctx.plan_path)
            or state['schedule_sha256']!=ctx.plan['schedule_sha256']
            or [r['run_id'] for r in state['runs']]!=[s['run_id'] for s in ctx.order]):
        raise ValueError('Expected complete frozen sweep')
    backends = state['backend_lifecycles']
    if len(backends)!=1 or backends[0]['role']!='base' or backends[0]['index']!=1:
        raise ValueError('Expected one common backend load')
    backend = backends[0]
    location = out/'private/backends/0001-base'
    served = read(location/'served-artifact.json')
    if (served['adapter'] or served['adapter_sha256']
            or served['profile_sha256']!=ctx.protocol['evaluation_profile_sha256']
            or read(location.with_name('0001-base-environment.json'))!=read(root/ctx.protocol['environment_path'])):
        raise ValueError('Unexpected model or environment')
    previous = module('audit_contract_035',root/'experiments/035-edit-bottleneck/audit.py')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ctx.runtime['model'],local_files_only=True,trust_remote_code=False)
    profile = read(root/ctx.protocol['evaluation_profile_path'])
    norm = normalization(root)
    journal = RequestJournal(location/'requests')
    reconstruction = out/'private/audit'
    reconstruction.mkdir()
    tasks = variants(root,ctx,proof)
    receipt = {r['task_number']:r for r in proof['rows']}
    existing_workspace = {r['task_number']:r for r in legacy['rows']}
    stuck = set(legacy['diagnosis']['stuck_tasks'])
    admission = {r['id']:r for r in read(ctx.data/'admission.json')['rows']}
    seen = set()
    rows = []
    beginnings = defaultdict(dict)
    workspaces = {}
    for item,result in zip(ctx.order,state['runs']):
        n,arm,run_id = (item[k] for k in ('task_number','arm','run_id'))
        task = tasks[run_id]
        source = ctx.sources[task['source_id']]
        if (result['backend_index']!=1 or result['id']!=task['id'] or result['task_number']!=n or result['contrast_arm']!=arm):
            raise ValueError('Misbound scored task')
        directory = out/'private/tasks'/run_id
        trace = read(directory/'trajectory.json')
        if trace['backend']!=served:
            raise ValueError('Trajectory and backend differ')
        if n not in workspaces:
            workspaces[n] = materialize(ctx.data/'snapshots'/task['source_id'],reconstruction/'broken'/f't{n:02d}',source,task,hidden=False)
        before = workspaces[n]
        if fingerprint(sorted(before.items()))!=existing_workspace[n]['workspace_sha256']:
            raise ValueError('Original public regression was not obtained from the identical workspace')
        row,start,calls = previous.audited_row(
            root,ctx.data,directory,result,task,source,ctx.protocol,profile,tokenizer,norm,
            journal,reconstruction/'graded'/run_id,before,backend)
        stored = receipt[n]
        if start!=stored['opening_token_sha256'][arm] or trace['steps'][0]['generation']['input_tokens']!=stored['opening_tokens'][arm]:
            raise ValueError('Actual opening tokenization differs from preregistration')
        if seen & calls:
            raise ValueError('Duplicate physical generation')
        seen.update(calls)
        beginnings[n][arm]=start
        row.update(multifile=admission[task['id']]['multifile'],old_prefix_clipped=stored['raw_prefix_clipped'],
                   recovered_failure_line=stored['selected_missing_failure_visible']>0,historically_stuck=n in stuck,
                   packet_chars=stored[arm+'_chars'],opening_input_tokens=stored['opening_tokens'][arm])
        rows.append(row)
    if seen!={p.name for p in journal.directory.iterdir() if p.is_dir()}:
        raise ValueError('Physical generation inventory differs')
    changed = sum(len(set(pair.values()))==2 for pair in beginnings.values())
    if changed!=len(ctx.tasks)-ctx.plan['preflight']['fallback']:
        raise ValueError('Expected actual opening change for every extracted prompt')
    summary = {
        'schema_version':1,'passed':True,'optimizer_updates':0,'final_test_opened':False,
        'registration_sha256':sha256(ctx.plan_path),'controller_sha256':sha256(path),
        'verified_requests':len(seen),'opening_change_pairs':changed,'backend_loads':len(backends),
        'started_epoch':state['started_epoch'],'finished_epoch':state['finished_epoch'],
        'telemetry':state['telemetry'],'preflight':ctx.plan['preflight'],
        'analysis':analyze(rows,ctx.plan['followup_gate'],len(ctx.tasks)),
        'arms':previous.totals(rows),'rows':rows,
    }
    write_json(out/'audit.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.root.resolve())
    print(json.dumps({'passed':result['passed'],'verified_requests':result['verified_requests'],
                      'opening_change_pairs':result['opening_change_pairs']}))
