"""Safe, reproducible training-data collection outcomes and feasibility decision."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from one_gpu_lm.common.receipts import write_json
from .recovery_report import hardware


def summarize(audit,protocol):
    rows=audit['rows'];by={(r['task_id'],r['arm']):r for r in rows}
    ids=protocol['diagnostic_task_ids']+protocol['supplemental_task_ids']
    complete=[name for name in ids if all(by.get((name,arm),{}).get('audit_passed') and by[(name,arm)]['complete'] for arm in ['control','correction'])]
    outcomes={}
    for arm in ['control','correction']:
        rr=[by[(name,arm)] for name in complete];qq=[r for r in rr if r['qualified']]
        outcomes[arm]={'success':sum(r['result']['success_within_budget'] for r in rr),'qualified':len(qq),
            'no_edit':sum(not r['encoding']['edit_calls'] for r in rr),
            'edited_failure':sum(bool(r['encoding']['edit_calls']) and not r['result']['success_within_budget'] for r in rr),
            'qualified_families':len({r['family'] for r in qq}),
            'qualified_supplemental_tasks':sum(r['cohort']=='supplemental' for r in qq),
            'qualified_task_ids':[r['task_id'] for r in qq],
            'calls':sum(r['result']['attempted_calls'] for r in rr),
            'generated_tokens':sum(r['result']['generated_tokens'] for r in rr)}
    gains=[n for n in complete if by[(n,'correction')]['qualified'] and not by[(n,'control')]['qualified']]
    losses=[n for n in complete if by[(n,'control')]['qualified'] and not by[(n,'correction')]['qualified']]
    g=protocol['feasibility_gate'];guided=outcomes['correction'];baseline=outcomes['control']
    checks={'complete_audit':audit['passed'] and len(complete)==len(ids),
        'qualified_unique_tasks':guided['qualified']>=g['minimum_qualified_unique_tasks'],
        'qualified_training_families':guided['qualified_families']>=g['minimum_qualified_families'],
        'qualified_supplemental_tasks':guided['qualified_supplemental_tasks']>=g['minimum_qualified_supplemental_tasks'],
        'net_qualified_gain':len(gains)-len(losses)>=g['minimum_net_qualified_gain'],
        'no_net_repair_success_decline':guided['success']>=baseline['success']}
    return {'schema_version':1,'registered_tasks':len(ids),'registered_runs':len(protocol['schedule']),'complete_audited_pairs':len(complete),
        'outcomes':outcomes,'qualified_gain_task_ids':gains,'qualified_regression_task_ids':losses,'net_qualified_gain':len(gains)-len(losses),
        'feasibility_gate':g,'feasibility_checks':checks,'data_feasibility_gate_passed':all(checks.values()),
        'actual_cost':{'attempted_calls':sum(r['attempted_calls'] for r in audit['attempts']),
                       'generated_tokens':sum(r['generated_tokens'] for r in audit['attempts']),
                       'input_tokens':sum(r['input_tokens'] for r in audit['attempts']),
                       'actor_seconds':sum(r['actor_seconds'] for r in audit['attempts']),
                       'unknown_generation_cost_attempts':sum(r['unknown_generation_cost'] for r in audit['attempts'])},
        'private_export_summary':audit['private_export_summary'],'new_student_optimizer_updates':0,
        'final_test_opened':False,'weights_public':False,'independent_model_quality_evaluation':False}


def render(audit,protocol,execution,out):
    out=Path(out);s=summarize(audit,protocol);write_json(out/'summary.json',s)
    rows=audit['rows'];by={(r['task_id'],r['arm']):r for r in rows};pairs=[]
    for name in protocol['diagnostic_task_ids']+protocol['supplemental_task_ids']:
        if all(by.get((name,a),{}).get('audit_passed') and by[(name,a)]['complete'] for a in ['control','correction']):
            pairs.append([name,by[(name,'control')]['family'],by[(name,'control')]['cohort'],
                *[int(by[(name,a)]['result']['success_within_budget']) for a in ['control','correction']],
                *[int(by[(name,a)]['qualified']) for a in ['control','correction']],
                *[by[(name,a)]['result']['attempted_calls'] for a in ['control','correction']]])
    with (out/'pairs.csv').open('w',newline='') as stream:
        w=csv.writer(stream,lineterminator='\n');w.writerow(['id','family','cohort','control_success','correction_success','control_qualified','correction_qualified','control_calls','correction_calls']);w.writerows(pairs)
    c=s['outcomes']['control'];g=s['outcomes']['correction'];gate=s['data_feasibility_gate_passed']
    (out/'README.md').write_text(f'''# 027: failure-conditioned corrective sample collection

This is a training-data feasibility comparison, not independent model-quality evidence. Both arms use the same pinned base teacher, fresh broken workspaces and the same 18-call budget. The correction arm adds a generic workflow prompt, actor-visible progress feedback and a mandatory standalone regression after every source edit. Their combined effect is measured; this experiment does not isolate each component.

- Registered tasks: {s['registered_tasks']}; completed audited pairs: {s['complete_audited_pairs']}.
- Independent repair successes: control {c['success']}, correction {g['success']}.
- Qualified multi-edit corrective trajectories: control {c['qualified']}, correction {g['qualified']}.
- Correction arm qualified families: {g['qualified_families']}; qualified supplemental tasks: {g['qualified_supplemental_tasks']}.
- Qualified gains / regressions: {len(s['qualified_gain_task_ids'])} / {len(s['qualified_regression_task_ids'])}; net: {s['net_qualified_gain']}.
- Preregistered collection feasibility gate passed: **{gate}**.
- Actual physical model calls: {s['actual_cost']['attempted_calls']}; generated tokens: {s['actual_cost']['generated_tokens']}.

Qualification requires an actual source edit, a standalone failing visible regression on that edited state, another distinct source change, final standalone passing regression, explicit submission, independent fresh grading, intact protected files throughout, complete request/feedback/cost audit and untruncated encoding. Timeout, rejected requests, an untouched original fault, compound-command exit codes and two edits without intervening regression failure do not qualify. A source change alone is not a semantic proof of repair.

Only the final successful edit and its subsequent executed verification/submission actions are positive training targets. Failed edits remain in context. The actual input for the final edit must retain both the failed edit and its failure feedback; context truncation cannot silently remove them. No deliberate mistakes are injected to obtain additional edits.

![Outcomes and cumulative qualified samples](correction.png)

The right-hand curve is accumulation over fixed task order, not a training learning curve. Eight cases are selected from observed 026 failures; 24 additional training tasks are selected by fixed hash and repository round-robin, excluding all 026 tasks. Neither cohort is an independent development or final test set. The 18-call comparison must not be presented as a fair comparison against 026's 12-call results.

No student parameter update or final-test access occurs in this collection stage. A passed data gate only permits preparation of a separately frozen single-candidate training experiment; it does not establish model improvement or authorize weight publication. Raw source, tasks, prompts, commands, trajectories and weights remain private.

Numeric tables and figures regenerate with `python -m one_gpu_lm.verified.correction_report --results <directory> --protocol <execution-protocol.json>`. See `audit.json`, `execution.json`, `telemetry.csv` and `hardware.json` for recorded evidence. Hardware power is GPU telemetry, not wall-outlet electricity.
''')
    return s,pairs


def plot(summary,pairs,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for i,(arm,color) in enumerate([('control','#3563b8'),('correction','#c76c32')]):
        value=summary['outcomes'][arm]
        bars=axes[0].bar([x+(i-.5)*.35 for x in range(2)],[value['success'],value['qualified']],width=.35,color=color,label=arm)
        axes[0].bar_label(bars,padding=3)
        cumulative=[];total=0
        for row in pairs:total+=row[5+i];cumulative.append(total)
        axes[1].plot(range(1,len(pairs)+1),cumulative,label=arm,color=color,linewidth=2)
    axes[0].set_xticks([0,1],['Repair success','Qualified correction']);axes[0].set_ylabel('Complete paired tasks');axes[0].legend()
    axes[1].set_xlabel('Complete pair in registered order');axes[1].set_ylabel('Cumulative qualified samples');axes[1].legend()
    axes[1].set_title('Sample accumulation, not learning');axes[0].set_title('Same base teacher and budgets')
    fig.suptitle(f"027 training-only correction collection | {summary['complete_audited_pairs']} complete pairs")
    fig.savefig(Path(out)/'correction.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);a=p.parse_args()
    audit=json.loads((a.results/'audit.json').read_text());execution=json.loads((a.results/'execution.json').read_text());protocol=json.loads(a.protocol.read_text())
    summary,pairs=render(audit,protocol,execution,a.results);write_json(a.results/'hardware.json',hardware(a.results/'telemetry.csv',execution));plot(summary,pairs,a.results)
