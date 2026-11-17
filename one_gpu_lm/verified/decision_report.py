"""029 report from safe audited results, with a preregistered net-yield gate."""
import argparse
import csv
import json
from pathlib import Path
from one_gpu_lm.common.receipts import write_json
from .failure_report import summarize as previous_summary
from .recovery_report import hardware


def summarize(audit, protocol):
    result = previous_summary(audit, protocol)
    c, g = (result['outcomes'][a] for a in ['control', 'correction'])
    result['feasibility_checks']['minimum_net_qualified_gain'] = (
        g['qualified'] - c['qualified'] >= protocol['feasibility_gate']['minimum_net_qualified_gain'])
    result['data_feasibility_gate_passed'] = all(result['feasibility_checks'].values())
    by = {(r['task_id'], r['arm']): r for r in audit['rows']}
    pairs = [s['task_id'] for s in protocol['selected_states'] if all(
        by.get((s['task_id'], a), {}).get('audit_passed') for a in ['control', 'correction'])]
    result['qualification_gain_task_ids'] = [n for n in pairs if by[n,'correction']['qualified'] and not by[n,'control']['qualified']]
    result['qualification_regression_task_ids'] = [n for n in pairs if by[n,'control']['qualified'] and not by[n,'correction']['qualified']]
    result['control_policy'] = '028 correction procedure rerun on original admitted states'
    result['treatment_policy'] = protocol['continuation_policy']
    return result


def render(directory, protocol):
    out = Path(directory);a = json.loads((out/'audit.json').read_text());e = json.loads((out/'execution.json').read_text())
    s = summarize(a, protocol);write_json(out/'summary.json',s);write_json(out/'hardware.json',hardware(out/'telemetry.csv',e))
    by = {(r['task_id'],r['arm']):r for r in a['rows']};pairs=[]
    for state in protocol['selected_states']:
        name=state['task_id']
        if not all(by.get((name,arm),{}).get('audit_passed') for arm in ['control','correction']):continue
        pairs.append([name,state['family'],*[int(by[name,arm]['result']['success_within_budget']) for arm in ['control','correction']],
                      *[int(by[name,arm]['qualified']) for arm in ['control','correction']],
                      *[by[name,arm]['result']['attempted_calls'] for arm in ['control','correction']]])
    with (out/'pairs.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['id','family','control_success','correction_success','control_qualified','correction_qualified','control_calls','correction_calls']);w.writerows(pairs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for index,(arm,color) in enumerate([('control','#3563b8'),('correction','#c76c32')]):
        result=s['outcomes'][arm]
        bars=axes[0].bar([x+(index-.5)*.35 for x in range(2)],[result['success'],result['qualified']],width=.35,label=arm,color=color)
        axes[0].bar_label(bars,padding=3)
        total=0;values=[]
        for row in pairs:total+=row[4+index];values.append(total)
        axes[1].plot(range(1,len(pairs)+1),values,color=color,label=arm)
        counts=[]
        for call in range(1,protocol['evaluation_budget']['max_model_calls']+1):
            counts.append(sum(bool(by[row[0],arm]['encoding']['edit_calls']) and min(by[row[0],arm]['encoding']['edit_calls'])<=call for row in pairs))
        axes[2].step(range(1,len(counts)+1),counts,where='post',color=color,label=arm)
    axes[0].set_xticks([0,1],['Repair success','Qualified samples']);axes[0].set_ylabel('Paired tasks')
    axes[1].set_xlabel('Pair in registered order');axes[1].set_ylabel('Cumulative qualified samples')
    axes[2].set_xlabel('New model call');axes[2].set_ylabel('Tasks with a real source change')
    for axis in axes:axis.legend();axis.set_ylim(bottom=0)
    fig.suptitle('029 visible repair decisions | training-only collection, not model learning')
    fig.savefig(out/'repair-decisions.png',dpi=160);plt.close(fig)
    c,g=(s['outcomes'][arm] for arm in ['control','correction'])
    (out/'README.md').write_text(f'''# 029: visible repair decisions

Training-data collection only; no parameter update or final-test access.

| Metric | 028 procedure rerun | Decision procedure |
|---|---:|---:|
| Complete paired tasks | {s['complete_audited_pairs']} | {s['complete_audited_pairs']} |
| Independent repair success | {c['success']} | {g['success']} |
| Qualified corrective demonstrations | {c['qualified']} | {g['qualified']} |
| No new source change | {c['no_new_edit']} | {g['no_new_edit']} |
| Actual new calls | {c['calls']} | {g['calls']} |

Preregistered collection gate passed: **{s['data_feasibility_gate_passed']}**.

![Outcomes, sample yield and source-change timing](repair-decisions.png)

Both arms use the same admitted historical failure snapshots and the 028 correction guard. Control is a fresh rerun of 028 correction, not the 028 unguided control. Treatment adds a generic task-contract decision prompt and factual visible-state/call-budget feedback. No task-specific solution, hidden case, extra tool execution or automatic submission is supplied. Historical actions are context, not positive targets. Both arms receive identical new budgets; repeated and rejected actions consume those budgets.

Qualification requires a changed source, standalone visible verification, explicit submission, independent training-task grading, complete state/request/cost replay and untruncated targets retaining the historical failure. Only the final successful new edit and subsequent executed actions are exported privately.

These nine states are selected training failures already inspected in 028. This experiment does not measure independent repository generalization or candidate model improvement. A byte change can be a comment-only change; the third curve measures source changes, not semantic repairs. Cumulative curves are sample yield, not training loss. Raw source, histories, commands and weights remain private.

Regenerate with `python -m one_gpu_lm.verified.decision_report --results <directory> --protocol <execution-protocol.json>`. See [discussion.md](discussion.md) for interpretation, [audit.json](audit.json) for all paths and [hardware.json](hardware.json) for GPU-only sampled telemetry.
''')
    return s


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    args=p.parse_args();render(args.results,json.loads(args.protocol.read_text()))
