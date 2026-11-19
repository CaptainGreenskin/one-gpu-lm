"""Apply one registered edit checkpoint to the same base, and audit every request."""
import argparse
from collections import defaultdict
from dataclasses import replace
from functools import partial
import importlib
import json
from pathlib import Path
import subprocess

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_tasks import prepare
from one_gpu_lm.verified.rollout import normalization

import audit
from baseline import EXPERIMENT, OUTPUT as BASE_OUTPUT, PRIVATE, inputs as baseline_inputs, read, sweep
from edit_checkpoint import checkpoint


OUTPUT = Path('runs/historic035-reminder-001')
MODULES = ('resilient_rollout','historic_replay')


def reminder_inputs(root):
    original = baseline_inputs(root)
    path = root / EXPERIMENT / 'reminder-plan.json'
    plan = read(path)
    receipt = read(root / PRIVATE / 'reminder-publication.json')
    if (not receipt['anonymous_files_verified'] or receipt['ci_conclusion']!='success'
            or receipt['plan_sha256']!=sha256(path) or not original.legacy.frozen(root,plan['frozen_files'])):
        raise ValueError('Require exact published and passing reminder preregistration')
    proof = read(root/BASE_OUTPUT/'audit.json')
    if (not proof['passed'] or not proof['first_inputs_matched'] or proof['final_test_opened']
            or not read(root/PRIVATE/'baseline-pipeline.json')['complete']
            or proof['base_controller_sha256'] != sha256(root/BASE_OUTPUT/'controller.json')
            or receipt['baseline_report_commit'] != plan['baseline_report_commit']):
        raise ValueError('Use only the completed audited base comparison')
    if (plan['evaluation_budget']!=original.plan['evaluation_budget']
            or plan['agent_policy']!=original.plan['agent_policy']
            or plan['normalization']!=original.plan['normalization']
            or plan['recovery_accounting_seconds']!=original.plan['recovery_accounting_seconds']
            or plan['final_test_opened'] or plan['optimizer_updates']!=0):
        raise ValueError('Only the one-request checkpoint can differ')
    order = [dict(row, arm='reminder', run_id=f"t{row['task_number']:02d}-reminder") for row in original.order]
    if len(order)!=plan['rollouts'] or fingerprint(order)!=plan['schedule_sha256']:
        raise ValueError('Task order differs from the registered matched cohort')
    protocol = dict(original.protocol, evaluation_variant='reminder')
    return replace(original, plan_path=path, plan=plan, order=order, protocol=protocol), proof


def patched_module(root, plan, module, generate=False):
    directory = root/'one_gpu_lm/verified'
    source = directory/f'{module}.py'
    target = directory/f'_edit035_{module}.py'
    if generate:
        if target.exists():
            raise ValueError('Retain the first generated experimental entry')
        subprocess.run(['patch','-s','-F','0','-o',str(target),str(source),
                        str(root/EXPERIMENT/f'{module}.patch')],cwd=root,check=True)
    if sha256(target)!=plan['generated_source_sha256'][module]:
        raise ValueError('Generated evaluator differs from preregistered bytes')
    return importlib.import_module(f'one_gpu_lm.verified._edit035_{module}')


def validate_relation(old, new, old_directory, new_directory):
    """Compare histories through the unchanged prefix; later divergence is expected."""
    triggers = [s['call'] for s in new['steps'] if (s.get('edit_checkpoint') or {}).get('triggered')]
    if triggers not in ([], [5]):
        raise ValueError('Checkpoint appeared outside the single registered boundary')
    before = 4 if triggers else max(len(old['steps']),len(new['steps']))
    old_prefix, new_prefix = old['steps'][:before], new['steps'][:before]
    changed = len(old_prefix) != len(new_prefix)
    matched_input_different_output = 0
    for call, (a,b) in enumerate(zip(old_prefix,new_prefix),1):
        token_match = a['generation']['input_tokens_sha256']==b['generation']['input_tokens_sha256']
        if call == 1 and not token_match:
            raise ValueError('Different first input on the same task')
        ids_match = a['generation']['ids'] == b['generation']['ids']
        same_source = (a['source_before'],a.get('source_after')) == (b['source_before'],b.get('source_after'))
        same_feedback = (a.get('observation',{}).get('returncode'),a.get('normalization',{}).get('delivered_sha256')) == (
            b.get('observation',{}).get('returncode'),b.get('normalization',{}).get('delivered_sha256'))
        changed |= not token_match or not ids_match or not same_source or not same_feedback
        matched_input_different_output += token_match and not ids_match
    if triggers:
        control = read(old_directory/'request-005.json') if len(old['steps'])>=5 else None
        candidate = read(new_directory/'request-005.json')
        if control is None:
            changed = True
        else:
            if control['messages'][0] == candidate['messages'][0]:
                raise ValueError('Triggered checkpoint was not delivered')
            comparable = dict(candidate, messages=[control['messages'][0], *candidate['messages'][1:]])
            # Tool wall time is present in private audit metadata but never encoded for the model.
            def visible(messages):
                return [{**message, 'extra':{**message.get('extra', {}),
                         'tool':{k:v for k,v in message['extra']['tool'].items() if k!='seconds'}}}
                        if 'tool' in message.get('extra',{}) else message for message in messages]
            comparable['messages'] = visible(comparable['messages'])
            control['messages'] = visible(control['messages'])
            changed |= comparable != control
    return bool(triggers), changed, matched_input_different_output


def analyze(rows, plan):
    group = defaultdict(dict)
    for row in rows:
        number,arm = row['task_number'],row['arm']
        if not row['audit_passed'] or arm in group[number]:
            raise ValueError('Duplicate or unchecked matched result')
        group[number][arm]=row
    if set(group)!=set(range(1,21)) or any(set(pair)!={'base','reminder'} for pair in group.values()):
        raise ValueError('Expected every registered result')
    positive,negative,clean_positive,clean_negative = 0,0,0,0
    changes = defaultdict(int)
    for pair in group.values():
        base,candidate = pair['base'],pair['reminder']
        delta = int(candidate['completed_repair'])-int(base['completed_repair'])
        changes[base['family']] += delta
        positive += delta>0
        negative += delta<0
        if candidate['checkpoint_triggered'] and not candidate['unchanged_prompt_difference']:
            clean_positive += delta>0
            clean_negative += delta<0
    rule = plan['mechanism_gate']
    return {
        'paired_tasks':len(group),'improved':positive,'regressed':negative,'net_completed_repairs':positive-negative,
        'by_family':dict(sorted(changes.items())),
        'interpretable_triggered_improvements':clean_positive,
        'interpretable_triggered_regressions':clean_negative,
        'unchanged_prompt_different_tasks':sum(r['unchanged_prompt_difference'] for r in rows if r['arm']=='reminder'),
        'same_input_different_output_calls':sum(r['same_input_different_output'] for r in rows if r['arm']=='reminder'),
        'checkpoint_triggered':sum(r['checkpoint_triggered'] for r in rows if r['arm']=='reminder'),
        'mechanism_followup_gate_passed':positive-negative>=rule['minimum_net_completed_repairs']
                and sum(v>0 for v in changes.values())>=rule['minimum_improved_families']
                and negative<=rule['maximum_regressed_tasks']
                and clean_positive-clean_negative>=rule['minimum_net_completed_repairs'],
        'final_test_opened':False,'weight_publication_allowed':False,
    }


def audit_results(root, ctx, baseline):
    out = root/OUTPUT
    controller_path = out/'controller.json'
    state = read(controller_path)
    if (not state['complete'] or not state['frozen_inputs_unchanged'] or not state['backend_stopped']
            or state['final_test_opened'] or state['optimizer_updates']!=0
            or state['registration_sha256']!=sha256(ctx.plan_path)
            or state['schedule_sha256']!=ctx.plan['schedule_sha256']
            or [r['run_id'] for r in state['runs']] != [row['run_id'] for row in ctx.order]):
        raise ValueError('Complete registered reminder sweep required')
    backends = state['backend_lifecycles']
    if len(backends)!=1 or backends[0]['role']!='base' or backends[0]['index']!=1:
        raise ValueError('Same unadapted base must serve all reminders')
    backend = backends[0]
    directory = out/'private/backends/0001-base'
    health = read(directory/'served-artifact.json')
    if (health['adapter'] or health.get('adapter_sha256',{})
            or health['profile_sha256']!=ctx.protocol['evaluation_profile_sha256']
            or read(directory.with_name('0001-base-environment.json'))!=read(root/ctx.protocol['environment_path'])):
        raise ValueError('Backend model environment differs')
    journal = RequestJournal(directory/'requests')
    profile = read(root/ctx.protocol['evaluation_profile_path'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ctx.runtime['model'],local_files_only=True,trust_remote_code=False)
    replayer = patched_module(root,ctx.plan,'historic_replay')
    # Load the audit in this process only; the base phase keeps its frozen unmodified replayer.
    audit.replay = partial(replayer.replay, checkpoint=checkpoint)
    norm = normalization(root)
    old_raw = {r['task_number']:r for r in read(root/BASE_OUTPUT/'controller.json')['runs']}
    old_rows = {r['task_number']:r for r in baseline['rows'] if r['arm']=='base'}
    if len(old_raw)!=len(ctx.order) or len(old_rows)!=len(ctx.order):
        raise ValueError('Incomplete base cohort')
    seen = set()
    rows = []
    reconstruction = out/'private/audit'
    reconstruction.mkdir()
    for item,result in zip(ctx.order,state['runs']):
        number = item['task_number']
        task = ctx.tasks[item['task_id']]
        source = ctx.sources[task['source_id']]
        if result['task_number']!=number or result['contrast_arm']!='reminder' or result['backend_index']!=1:
            raise ValueError('Misbound reminder result')
        trial = out/'private/tasks'/item['run_id']
        old_dir = root/BASE_OUTPUT/'private/tasks'/old_raw[number]['run_id']
        if sha256(old_dir/'trajectory.json')!=old_raw[number]['trajectory_sha256']:
            raise ValueError('Base trajectory differs from its audited fingerprint')
        snapshot = ctx.data/'snapshots'/task['source_id']
        before = prepare(snapshot,reconstruction/'broken'/item['run_id'],source,task)
        row,first,requests = audit.audited_row(root,ctx.data,trial,result,task,source,ctx.protocol,profile,tokenizer,norm,
                journal,reconstruction/'graded'/item['run_id'],before,backend)
        if seen & requests:
            raise ValueError('Generation journal replayed more than once')
        seen.update(requests)
        old_trace = read(old_dir/'trajectory.json')
        trial_trace = read(trial/'trajectory.json')
        if old_trace['id']!=task['id'] or first!=old_trace['steps'][0]['generation']['input_tokens_sha256']:
            raise ValueError('Different first input on the same task')
        triggered, changed, unstable = validate_relation(old_trace,trial_trace,old_dir,trial)
        base = old_rows[number]
        row.update(multifile=base['multifile'], checkpoint_triggered=triggered,
                   unchanged_prompt_difference=changed, same_input_different_output=unstable)
        rows.extend([base,row])
    if {p.name for p in journal.directory.iterdir() if p.is_dir()}!=seen:
        raise ValueError('Unaccounted reminder generation')
    analysis = analyze(rows,ctx.plan)
    summary = {
        'schema_version':1,'passed':True,'registration_sha256':state['registration_sha256'],
        'controller_sha256':sha256(controller_path),'reminder_verified_generation_requests':len(seen),
        'base_verified_generation_requests':baseline['base_verified_generation_requests'],
        'first_inputs_matched':True,'optimizer_updates':0,'final_test_opened':False,
        'started_epoch':state['started_epoch'],'finished_epoch':state['finished_epoch'],
        'telemetry':state['telemetry'],'arms':audit.totals(rows),'rows':rows,'analysis':analysis,
    }
    write_json(out/'audit.json',summary)
    print(json.dumps({k:summary[k] for k in ('passed','reminder_verified_generation_requests')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--audit',action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    ctx,baseline = reminder_inputs(root)
    if args.check:
        print(json.dumps({'scheduled_rollouts':len(ctx.order),'final_test_opened':False,'cpu_preflight_passed':True}))
    elif args.audit:
        audit_results(root,ctx,baseline)
    else:
        actor = patched_module(root,ctx.plan,'resilient_rollout',generate=True)
        patched_module(root,ctx.plan,'historic_replay',generate=True)
        def run_with_checkpoint(*args,**kwargs):
            return actor.run(*args,**kwargs,checkpoint=checkpoint)
        sweep(root,root/OUTPUT,ctx,perform=run_with_checkpoint)


if __name__=='__main__':
    main()
