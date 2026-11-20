"""Run both registered formats on the same base, without changing any prior run."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.rollout import normalization

from prepare import ARMS, EXP, OLD, OUT, PARENT, PRIVATE, compact_counts, module, order_for, parent_inputs, present, read, text_hash


def inputs(root):
    runner,ctx,legacy = parent_inputs(root)
    path = root/EXP/'protocol.json'
    registration = read(path)
    publication = read(root/PRIVATE/'publication.json')
    if (publication['anonymous_files_verified'] is not True or publication['ci_conclusion']!='success'
            or publication['plan_sha256']!=sha256(path)
            or publication['prior_report_commit']!=registration['036_commit']
            or not ctx.legacy.frozen(root,registration['frozen_files'])):
        raise ValueError('Require exact frozen public protocol')
    keys = ('evaluation_budget','agent_policy','normalization','recovery_accounting_seconds','followup_gate')
    if (any(registration[k]!=ctx.plan[k] for k in keys)
            or registration['optimizer_updates']!=0 or registration['final_test_opened']
            or registration['weight_publication_allowed'] or ctx.protocol['adapter_sha256']):
        raise ValueError('Only the initial visible test output can differ')
    order = order_for(ctx)
    if len(order)!=registration['rollouts'] or fingerprint(order)!=registration['schedule_sha256']:
        raise ValueError('Task-pair schedule differs')
    proof = read(root/PRIVATE/'compaction-preflight.json')
    if (not proof['passed'] or proof['final_test_opened'] or proof['model_calls']!=0 or proof['new_tool_calls']!=0
            or proof['schedule_sha256']!=registration['schedule_sha256']
            or compact_counts(proof)!=registration['preflight']):
        raise ValueError('Require the audited CPU prompt and evidence preflight')
    return runner,replace(ctx,plan_path=path,plan=registration,order=order),proof,legacy


def variants(root, ctx, proof):
    evidence = module('prior036_excerpt_inputs',root/PARENT/'evidence.py')
    norm = normalization(root)
    cap = ctx.protocol['evaluation_budget']['max_tool_output_chars']
    rows = {row['task_number']:row for row in proof['rows']}
    result = {}
    for i in range(0,len(ctx.order),2):
        pair = ctx.order[i:i+2]
        n = pair[0]['task_number']
        task = ctx.tasks[pair[0]['task_id']]
        path = root/OLD/f'public-regressions/t{n:02d}/output.json'
        row = rows[n]
        if sha256(path)!=row['transcript_sha256']:
            raise ValueError('The visible test output changed since preflight')
        prefix,compact,stats = present(task,read(path),norm,cap,evidence.make_task)
        if any(row[k]!=v for k,v in stats.items()):
            raise ValueError('Compaction metadata differs from frozen preflight')
        for item in pair:
            selected = prefix if item['arm']=='prefix' else compact
            if (item['task_id']!=selected['id']
                    or text_hash(selected['description'])!=row['description_sha256'][item['arm']]):
                raise ValueError('Task prompt differs')
            result[item['run_id']] = selected
    return result


def run(root, runner, ctx, tasks):
    def perform(root, data, task, source, backend, sandbox, protocol, directory, arm, **kwargs):
        selected = tasks[directory.name]
        if selected['id']!=task['id']:
            raise ValueError('Misbound scheduled task')
        return runner.rollout(root,data,selected,source,backend,sandbox,protocol,directory,arm,**kwargs)
    return runner.sweep(root,root/OUT,ctx,perform=perform)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    runner,ctx,proof,_ = inputs(root)
    tasks = variants(root,ctx,proof)
    if args.check:
        print(json.dumps({'tasks':len(ctx.tasks),'rollouts':len(tasks),'final_test_opened':False}))
    else:
        run(root,runner,ctx,tasks)
