"""Run both public-regression evidence arms with frozen budgets and one base load."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_admit import semantic_failure
from one_gpu_lm.verified.historic_policy import feedback
from one_gpu_lm.verified.rollout import normalization

from evidence import ARMS, EXP, OUTPUT, PRIVATE, baseline, make_task, read, schedule


def inputs(root):
    parent = baseline(root)
    ctx = parent.inputs(root)
    path = root / EXP / 'protocol.json'
    plan = read(path)
    published = read(root / PRIVATE / 'publication.json')
    if (published['anonymous_files_verified'] is not True or published['ci_conclusion'] != 'success'
            or published['plan_sha256'] != sha256(path)
            or not ctx.legacy.frozen(root,plan['frozen_files'])):
        raise ValueError('Exact frozen publication and CI must match')
    if any(plan[key] != ctx.plan[key] for key in
           ('evaluation_budget','agent_policy','normalization','recovery_accounting_seconds')):
        raise ValueError('036 must use the 035 model and full per-task budget')
    if ([ctx.protocol[k] for k in ('model_id','revision')] != [plan[k] for k in ('model_id','revision')]
            or ctx.protocol['adapter_sha256'] or plan['optimizer_updates']!=0
            or plan['final_test_opened'] or plan['weight_publication_allowed']
            or plan['evaluation_cohort']!='034_development'):
        raise ValueError('Use only the pinned unadapted base on the frozen development cohort')
    candidates = schedule(ctx)
    if len(candidates)!=plan['rollouts'] or fingerprint(candidates)!=plan['schedule_sha256']:
        raise ValueError('Wrong frozen run order')
    proof = read(root / PRIVATE / 'evidence-preflight.json')
    if (not proof['passed'] or proof['model_calls']!=0 or proof['final_test_opened']
            or proof['schedule_sha256']!=plan['schedule_sha256']
            or proof['public_regressions']!=len(ctx.tasks) or len(proof['rows'])!=len(ctx.tasks)):
        raise ValueError('Require complete score-blind public regression preflight')
    evidence_rows = {r['task_number']:r for r in proof['rows']}
    normalizer = normalization(root)
    cap = plan['evaluation_budget']['max_tool_output_chars']
    for row in ctx.order:
        number = row['task_number']
        task = ctx.tasks[row['task_id']]
        if task['split']!='development':
            raise ValueError('Expected development tasks only')
        output = root / PRIVATE / f'public-regressions/t{number:02d}/output.json'
        value = read(output)
        measure = evidence_rows[number]
        transcript, receipt = feedback(value['output'],normalizer,cap)
        if (not semantic_failure(value) or sha256(output)!=measure['output_sha256']
                or value['returncode']!=measure['returncode'] or receipt!=measure['normalization']
                or len(transcript)!=measure['transcript_chars']):
            raise ValueError('Stored visible regression output changed')
    return parent, replace(ctx, plan_path=path,plan=plan,order=candidates), proof


def tasks_with_evidence(root, ctx):
    """Share immutable tool output but copy task descriptions per arm."""
    normalizer = normalization(root)
    cap = ctx.plan['evaluation_budget']['max_tool_output_chars']
    result = {}
    for item in ctx.order:
        number = item['task_number']
        output = read(root / PRIVATE / f'public-regressions/t{number:02d}/output.json')
        task = ctx.tasks[item['task_id']]
        result[item['run_id']] = make_task(task,output,item['arm'],normalizer,cap)
    return result


def run(root, parent, ctx):
    task_by_run = tasks_with_evidence(root, ctx)

    def perform(root, data, task, source, backend, sandbox, protocol, directory, arm, **kwargs):
        selected = task_by_run[directory.name]
        if selected['id'] != task['id']:
            raise ValueError('Misbinding of registered task and visible evidence')
        return parent.rollout(root,data,selected,source,backend,sandbox,protocol,directory,arm,**kwargs)

    return parent.sweep(root,root/OUTPUT,ctx,perform=perform)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    parent,ctx,proof = inputs(root)
    if args.check:
        print(json.dumps({'tasks':len(ctx.tasks),'rollouts':len(ctx.order),
                          'cpu_public_checks':proof['public_regressions'],'final_test_opened':False}))
    else:
        run(root,parent,ctx)
