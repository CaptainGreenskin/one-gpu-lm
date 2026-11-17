"""Reproducible 028 state-conditioned collection results; no model-quality claim."""
import argparse
import csv
import json
from pathlib import Path
from one_gpu_lm.common.receipts import write_json
from .recovery_report import hardware


def summarize(audit,p):
    by={(r['task_id'],r['arm']):r for r in audit['rows']};ids=[r['task_id'] for r in p['selected_states']]
    complete=[n for n in ids if all(by.get((n,a),{}).get('audit_passed') and by[(n,a)]['complete'] for a in ['control','correction'])]
    outcomes={}
    for arm in ['control','correction']:
        rows=[by[(n,arm)] for n in complete];qualified=[r for r in rows if r['qualified']]
        outcomes[arm]={'success':sum(r['result']['success_within_budget'] for r in rows),'qualified':len(qualified),
            'qualified_families':len({r['family'] for r in qualified}),
            'qualified_not_previously_qualified_027':sum(not r['previously_qualified_027'] for r in qualified),
            'qualified_task_ids':[r['task_id'] for r in qualified],
            'no_new_edit':sum(not r['encoding']['edit_calls'] for r in rows),
            'edited_failure':sum(bool(r['encoding']['edit_calls']) and not r['result']['success_within_budget'] for r in rows),
            'calls':sum(r['result']['attempted_calls'] for r in rows),
            'generated_tokens':sum(r['result']['generated_tokens'] for r in rows),
            'actor_seconds':sum(r['result']['actor_seconds'] for r in rows)}
    c=outcomes['control'];g=outcomes['correction'];gate=p['feasibility_gate']
    checks={'complete_audit':audit['passed'] and len(complete)==len(ids),
        'qualified_unique_tasks':g['qualified']>=gate['minimum_qualified_unique_tasks'],
        'qualified_families':g['qualified_families']>=gate['minimum_qualified_families'],
        'additional_corrective_tasks_beyond_027':g['qualified_not_previously_qualified_027']>=gate['minimum_additional_corrective_tasks_beyond_027'],
        'no_net_qualified_decline':g['qualified']>=c['qualified'],
        'no_net_repair_success_decline':g['success']>=c['success']}
    gains=[n for n in complete if by[n,'correction']['result']['success_within_budget'] and not by[n,'control']['result']['success_within_budget']]
    losses=[n for n in complete if by[n,'control']['result']['success_within_budget'] and not by[n,'correction']['result']['success_within_budget']]
    return {'schema_version':1,'registered_tasks':len(ids),'registered_runs':len(p['schedule']),'complete_audited_pairs':len(complete),
        'outcomes':outcomes,'repair_gain_task_ids':gains,'repair_regression_task_ids':losses,
        'feasibility_checks':checks,'data_feasibility_gate_passed':all(checks.values()),
        'actual_new_cost':{k:sum(r[k] for r in audit['attempts']) for k in ['attempted_calls','generated_tokens','input_tokens','actor_seconds']},
        'unknown_generation_cost_attempts':sum(r['unknown_generation_cost'] for r in audit['attempts']),
        'historical_prefix_cost_reused_once_per_task':{k:sum(by[n,'control']['historical_prefix_cost'][k] for n in complete) for k in ['historical_calls','historical_generated_tokens','historical_input_tokens']},
        'private_export_summary':audit['private_export_summary'],'new_student_optimizer_updates':0,
        'final_test_opened':False,'weights_public':False,'independent_model_quality_evaluation':False}


def render(audit,p,execution,out):
    out=Path(out);s=summarize(audit,p);write_json(out/'summary.json',s);write_json(out/'hardware.json',hardware(out/'telemetry.csv',execution))
    by={(r['task_id'],r['arm']):r for r in audit['rows']};pairs=[]
    for state in p['selected_states']:
        name=state['task_id']
        if all(by.get((name,a),{}).get('audit_passed') and by[name,a]['complete'] for a in ['control','correction']):
            pairs.append([name,state['family'],state['state_id'],*[int(by[name,a]['result']['success_within_budget']) for a in ['control','correction']],
                *[int(by[name,a]['qualified']) for a in ['control','correction']],*[by[name,a]['result']['attempted_calls'] for a in ['control','correction']]])
    with (out/'pairs.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['id','family','state_id','control_success','correction_success','control_qualified','correction_qualified','control_calls','correction_calls']);w.writerows(pairs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for i,(arm,color) in enumerate([('control','#3563b8'),('correction','#c76c32')]):
        values=s['outcomes'][arm]
        bars=axes[0].bar([x+(i-.5)*.35 for x in range(2)],[values['success'],values['qualified']],width=.35,color=color,label=arm)
        axes[0].bar_label(bars,padding=3);cumulative=[];total=0
        for row in pairs:total+=row[5+i];cumulative.append(total)
        axes[1].plot(range(1,len(pairs)+1),cumulative,label=arm,color=color,linewidth=2)
    axes[0].set_xticks([0,1],['Repair success','Qualified correction']);axes[0].set_ylabel('Complete paired tasks');axes[0].legend()
    axes[1].set_xlabel('Complete pair in registered order');axes[1].set_ylabel('Cumulative qualified samples');axes[1].legend()
    axes[0].set_title('Same failure state and new budget');axes[1].set_title('Sample accumulation, not learning')
    fig.suptitle(f"028 training-only failed-state continuations | {s['complete_audited_pairs']} complete pairs")
    fig.savefig(out/'failure-recovery.png',dpi=160);plt.close(fig)
    c=s['outcomes']['control'];g=s['outcomes']['correction']
    (out/'README.md').write_text(f'''# 028: continuations from admitted historical failure states

Training-data feasibility only. No model parameter update or final-test access.

- Complete audited pairs: {s['complete_audited_pairs']} / {s['registered_tasks']}.
- Repair success: control {c['success']}, correction {g['success']}.
- Qualified corrective continuations: control {c['qualified']}, correction {g['qualified']}.
- Correction qualified families: {g['qualified_families']}; qualified task IDs beyond the previously qualified 027 task: {g['qualified_not_previously_qualified_027']}.
- Preregistered data gate passed: **{s['data_feasibility_gate_passed']}**.
- Actual new model calls: {s['actual_new_cost']['attempted_calls']}; generated tokens: {s['actual_new_cost']['generated_tokens']}.

![Outcomes and cumulative corrective samples](failure-recovery.png)

Both branches restore identical newly reconstructed snapshots and full visible historical prefixes. A recorded standalone visible revalidation confirmed that each modified state still fails. The historical system message is retained in both branches; one prefix already includes 027 guidance. Control adds only a shared continuation-budget and validation notice. Correction additionally adds generic guidance and requires a standalone visible regression after every new source edit. This is a bundled continuation-policy comparison, not isolated component causality.

Each arm receives 12 new calls, at most 512 generated tokens per call and 6144 in total, 8192 input tokens and 600 actor seconds. Historical generation costs are reported separately and are not new calls. Snapshot reconstruction preserves current content/modes/mtime for equal forks, not the old inode, external clock or process memory.

Qualification requires a new source correction, final standalone passing regression, explicit submission, fresh independent training-task grading, protected files and complete request/state/cost replay. The final successful edit input must still contain the actual historical failed-edit action and the recorded revalidation feedback. Only the final successful new edit and subsequent executed actions are positive targets; old actions and earlier new failed edits are context only. Whole continuations with missing context or truncated targets are rejected.

These are selected training failures, possibly previously used training tasks. They do not establish independent model quality or long-task competence. A passed collection gate permits a separately frozen candidate training/development stage, not weight release.

Regenerate numeric outputs and the plot with `python -m one_gpu_lm.verified.failure_report --results <directory> --protocol <execution-protocol.json>`. Raw tasks, source, commands, histories and weights remain private. GPU telemetry is not wall-outlet electricity. The curve is cumulative sample yield, not a training learning curve.
''')
    return s


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);parser.add_argument('--protocol',type=Path,required=True);args=parser.parse_args()
    render(json.loads((args.results/'audit.json').read_text()),json.loads(args.protocol.read_text()),json.loads((args.results/'execution.json').read_text()),args.results)
