"""Allowlisted, reproducible paired-pilot report and figures; no raw trajectories."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import statistics
from one_gpu_lm.common.receipts import write_json


def cluster_interval(pairs, draws=20000):
    groups = defaultdict(list)
    for row in pairs:
        groups[row['family']].append(int(row['teacher']['success_within_budget'])-int(row['student']['success_within_budget']))
    if not groups: return None
    names = sorted(groups); rng = random.Random(2601); scores = []
    for _ in range(draws):
        sample = [v for name in rng.choices(names, k=len(names)) for v in groups[name]]
        scores.append(100*sum(sample)/len(sample))
    scores.sort()
    return {'method':'repository-cluster percentile bootstrap; descriptive, few clusters', 'clusters':len(groups),
            'draws':draws, 'seed':2601, 'lower_pp':scores[int(.025*draws)], 'upper_pp':scores[int(.975*draws)-1]}


def summarize(audit, execution):
    pairs = [r for r in audit['rows'] if r['complete'] and r['audit_passed'] and r['disposition']=='paired']
    gains = [r['id'] for r in pairs if r['teacher']['success_within_budget'] and not r['student']['success_within_budget']]
    losses = [r['id'] for r in pairs if r['student']['success_within_budget'] and not r['teacher']['success_within_budget']]
    qualified = [r for r in pairs if r['qualified_teacher_continuation']]
    outcomes = {}
    for branch in ['student','teacher']:
        outcomes[branch] = {'success':sum(r[branch]['success_within_budget'] for r in pairs),
            'no_edit':sum(r[branch]['first_edit_call'] is None for r in pairs),
            'edited_failure':sum(not r[branch]['success_within_budget'] and r[branch]['first_edit_call'] is not None for r in pairs),
            'calls':sum(r[branch]['attempted_calls'] for r in pairs),
            'generated_tokens':sum(r[branch]['generated_tokens'] for r in pairs),
            'actor_seconds':sum(r[branch]['actor_seconds'] for r in pairs)}
    gate = {'all_attempts_audited':audit['passed'], 'twelve_qualified_unique_continuations':len(qualified)>=12,
            'three_training_families':len({r['family'] for r in qualified})>=3, 'net_paired_success_gain_at_least_three':len(gains)-len(losses)>=3}
    return {'schema_version':1, 'registered_training_tasks':len(audit['rows']),
            'task_dispositions':dict(Counter(r['disposition'] for r in audit['rows'])), 'complete_audited_pairs':len(pairs),
            'paired_outcomes':outcomes, 'teacher_gain_task_ids':gains, 'teacher_regression_task_ids':losses,
            'net_teacher_gain':len(gains)-len(losses),
            'paired_success_difference_pp':100*(len(gains)-len(losses))/len(pairs) if pairs else None,
            'cluster_interval':cluster_interval(pairs), 'qualified_unique_teacher_continuations':len(qualified),
            'qualified_training_families':len({r['family'] for r in qualified}), 'data_feasibility_checks':gate,
            'data_feasibility_gate_passed':all(gate.values()), 'private_export_summary':audit['private_export_summary'],
            'actual_cost_shared_prefix_once':{
                'attempted_calls':sum(a['attempted_calls'] for a in audit['attempts']),
                'known_generated_tokens':sum(a['known_generated_tokens'] for a in audit['attempts']),
                'known_input_tokens':sum(a['known_input_tokens'] for a in audit['attempts']),
                'actor_seconds':sum(a['actor_seconds'] for a in audit['attempts']),
                'unknown_inflight_requests':sum(a['unknown_inflight_generation_cost'] for a in audit['attempts'])},
            'execution':execution, 'new_student_optimizer_updates':0, 'independent_development_evaluation':False,
            'final_test_opened':False, 'weights_public':False, 'model_improvement_claim_allowed':False}


def hardware(path, execution):
    with Path(path).open() as stream: rows=list(csv.DictReader(stream))
    times=[datetime.fromisoformat(r['utc']).timestamp() for r in rows]
    gaps=[b-a for a,b in zip(times,times[1:])]
    duration=execution['finished_epoch']-execution['started_epoch']
    result={'samples':len(rows), 'collector':execution['telemetry'], 'interval_seconds':5,
            'first_to_last_seconds':times[-1]-times[0] if times else 0,
            'maximum_gap_seconds':max(gaps,default=0), 'pipeline_seconds':duration,
            'first_sample_delay_seconds':times[0]-execution['started_epoch'] if times else None,
            'last_sample_to_finish_seconds':execution['finished_epoch']-times[-1] if times else None}
    for key in ['temperature_c','power_w','utilization_percent','memory_used_mib']:
        values=[float(r[key]) for r in rows]
        result[key]={'mean':statistics.mean(values), 'maximum':max(values)} if values else None
    result['sampled_gpu_energy_kwh']=sum((float(a['power_w'])+float(b['power_w']))/2*gap for a,b,gap in zip(rows,rows[1:],gaps))/3600000
    result['energy_scope']='GPU-only sampled trapezoid estimate, includes loading; not wall electricity'
    return result


def render(audit, execution, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); summary=summarize(audit,execution)
    write_json(out/'summary.json',summary)
    pairs=[r for r in audit['rows'] if r['complete'] and r['audit_passed'] and r['disposition']=='paired']
    columns=['id','family','student_success','teacher_success','student_calls','teacher_calls','student_tokens','teacher_tokens','qualified_teacher']
    with (out/'pairs.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(columns)
        for r in pairs:
            writer.writerow([r['id'],r['family'],int(r['student']['success_within_budget']),int(r['teacher']['success_within_budget']),
                r['student']['attempted_calls'],r['teacher']['attempted_calls'],r['student']['generated_tokens'],r['teacher']['generated_tokens'],int(r['qualified_teacher_continuation'])])
    outcome=summary['paired_outcomes']; n=len(pairs); interval=summary['cluster_interval']
    ci=f"{interval['lower_pp']:.2f} to {interval['upper_pp']:.2f} pp ({interval['clusters']} repository clusters)" if interval else 'unavailable'
    text=f'''# 026: live-state recovery pilot

This is a training-data feasibility experiment, not a new model-quality evaluation. Tasks may overlap the 025 student's training. No student parameter update or final-test access occurred.

- Registered training tasks: {summary['registered_training_tasks']}; complete audited pairs: {n}.
- Student successes: {outcome['student']['success']}/{n}; base teacher successes after student prefix: {outcome['teacher']['success']}/{n}.
- Teacher gains / regressions: {len(summary['teacher_gain_task_ids'])} / {len(summary['teacher_regression_task_ids'])}; net gain: {summary['net_teacher_gain']}.
- Qualified unique teacher continuations: {summary['qualified_unique_teacher_continuations']} across {summary['qualified_training_families']} training families.
- Preregistered data-feasibility gate passed: **{summary['data_feasibility_gate_passed']}**. This gate does not authorize model or weight release.
- Repository-cluster bootstrap interval: {ci}. Five available families and deterministic synthetic repair tasks limit generalization; this is descriptive uncertainty, not a significance claim.

Both branches restore the same live state after three student calls. No edit at call three is a sampling rule, not proof that the student is stuck. Each receives at most nine additional calls, with generated tokens and actor time deducted from the same prefix. Actual collection cost charges that prefix once. Backend loading is excluded from actor caps and included in pipeline duration. Early edits, early completion, nonserializable states and incomplete attempts are retained in the audit; only completed eligible pairs enter the paired comparison.

The snapshot preserves file contents, empty directories, modes and mtime plus the complete visible history. Inode, ctime, atime, external clock and process memory are not preserved. No hidden/reference files are mounted in the actor. Identical snapshots do not imply perfectly identical hardware timing.

Positive continuation targets include every actually executed teacher action in a qualified continuation, including earlier failed commands. Malformed outputs and policy-rejected unexecuted requests remain in history and exclusion counts. Student prefix actions are never teacher targets. An encoding failure rejects the whole continuation; targets are not silently truncated.

![Paired outcomes and cumulative successes](recovery.png)

The cumulative plot follows the preregistered task order and is **not a training learning curve**. Raw prompts, commands, tasks, source documents, credentials and weights are not published. `audit.json`, `execution.json` and `telemetry.csv` regenerate the numeric report with `python -m one_gpu_lm.verified.recovery_report --results <directory>`.
'''
    (out/'README.md').write_text(text)
    return summary


def plot(audit, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pairs=[r for r in audit['rows'] if r['complete'] and r['audit_passed'] and r['disposition']=='paired']
    summary=summarize(audit,{}); fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    colors={'student':'#3563b8','teacher':'#c76c32'}
    for i,branch in enumerate(['student','teacher']):
        y=[summary['paired_outcomes'][branch][k] for k in ['success','no_edit','edited_failure']]
        bars=axes[0].bar([x+(i-.5)*.35 for x in range(3)],y,width=.35,color=colors[branch],label=branch)
        axes[0].bar_label(bars,padding=3)
        cumulative=[]; total=0
        for r in pairs: total+=int(r[branch]['success_within_budget']);cumulative.append(total)
        axes[1].plot(range(1,len(pairs)+1),cumulative,label=branch,color=colors[branch],linewidth=2)
    axes[0].set_xticks(range(3),['Success','No edit','Edited failure']);axes[0].set_ylabel('Complete paired tasks')
    axes[0].set_title('Same state, same remaining budget');axes[0].legend()
    axes[1].set_xlabel('Complete pair in registered order');axes[1].set_ylabel('Cumulative successes')
    axes[1].set_title('Descriptive accumulation, not learning');axes[1].legend()
    fig.suptitle(f'026 training-only recovery pilot | {len(pairs)} complete pairs')
    fig.savefig(Path(out)/'recovery.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);args=parser.parse_args()
    audit=json.loads((args.results/'audit.json').read_text());execution=json.loads((args.results/'execution.json').read_text())
    render(audit,execution,args.results);write_json(args.results/'hardware.json',hardware(args.results/'telemetry.csv',execution));plot(audit,args.results)
