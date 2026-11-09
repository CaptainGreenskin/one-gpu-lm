"""Paired, identity-checked analysis of one frozen coding quality screen."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import hashlib
import random
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.evaluation.report import paired


def assess(base,adapted,protocol,legacy_base,legacy_adapted,training):
    summary=paired(base,adapted,protocol['evaluation_task_counts'])
    if not legacy_base['complete'] or not legacy_adapted['complete'] or legacy_base['task_sha256']!=legacy_adapted['task_sha256']:
        raise ValueError('Unpaired legacy results')
    ids=[r['id'] for r in legacy_base['results']]
    if not ids or len(ids)!=len(set(ids)) or ids!=[r['id'] for r in legacy_adapted['results']]:
        raise ValueError('Legacy result membership changed')
    if training['status']!='completed' or training['steps']!=protocol['execution']['max_optimizer_updates']:
        raise ValueError('Training did not finish the frozen budget')
    gate=protocol['promotion'];fresh=summary['splits']['test'];historical=summary['splits']['dev']
    checks={'fresh_net_gain':fresh['adapted_passed']-fresh['baseline_passed']>=gate['minimum_net_fresh_holdout_gain'],
        'no_fresh_regressions':len(fresh['regressed'])<=gate['maximum_regressed_fresh_tasks'],
        'no_historical_regressions':len(historical['regressed'])<=gate['maximum_regressed_historical_tasks'],
        'all_legacy_functions_pass':sum(r['passed'] for r in legacy_adapted['results'])==len(ids)==gate['legacy_function_passes_required'],
        'adapter_reload_exact':training['saved_reload_greedy_ids_equal']}
    per_repo={}
    for repository in sorted({r['repository'] for r in summary['pairs']}):
        rows=[r for r in summary['pairs'] if r['repository']==repository]
        per_repo[repository]={'pairs':len(rows),'role':'fresh_holdout' if rows[0]['split']=='test' else 'historical_regression',
            'base_passed':sum(r['baseline_passed'] for r in rows),'adapted_passed':sum(r['adapted_passed'] for r in rows),
            'improved':[r['id'] for r in rows if not r['baseline_passed'] and r['adapted_passed']],
            'regressed':[r['id'] for r in rows if r['baseline_passed'] and not r['adapted_passed']]}
    summary.update(schema_version=1,gate=checks,advance_to_replication=all(checks.values()),daily_model_accepted=False,
        per_repository=per_repo,legacy={'base_passed':sum(r['passed'] for r in legacy_base['results']),
            'adapted_passed':sum(r['passed'] for r in legacy_adapted['results']),'tasks':len(ids)},
        failures={name:dict(Counter(r['exit_status'] for r in value['results'])) for name,value in [('base',base),('adapted',adapted)]})
    return summary


def render(root):
    read=lambda name:json.loads((root/name).read_text())
    protocol=read('protocol.json');base=read('baseline.json');adapted=read('adapted.json');training=read('training/result.json')
    controller=read('controller.json');selection=read('data-selection.json')
    if not controller['complete'] or controller['protocol_sha256']!=sha256(root/'protocol.json'):
        raise ValueError('Controller incomplete or changed protocol')
    if base['identity']['protocol_sha256']!=sha256(root/'protocol.json') or base['identity']['admission_sha256']!=sha256(root/'admission.json'):
        raise ValueError('Evaluation protocol/admission identity mismatch')
    if base['identity']['tasks_sha256']!=protocol['task_spec_sha256']:
        raise ValueError('Tasks differ from frozen protocol')
    if sha256(root/'data-selection.json')!=protocol['frozen_files']['runs/prepared-data/selection.json'] or selection['data_sha256']!=protocol['prepared_data_sha256']:
        raise ValueError('Selection differs from frozen training inputs')
    identity=read('training/identity.json');trainer_selection=read('training/selection.json')
    if sha256(root/'profile.json')!=identity['input_hashes']['profile'] or sha256(root/'data-receipt.json')!=selection['receipt_sha256']:
        raise ValueError('Published profile or data receipt changed')
    for sources in [base['identity']['source_files'],identity['source_files']]:
        if any(protocol['frozen_files'].get(name)!=digest for name,digest in sources.items()):
            raise ValueError('Execution source differs from frozen protocol')
    if identity['input_hashes']['data']!=selection['data_sha256'] or identity['input_hashes']['data_receipt']!=selection['receipt_sha256']:
        raise ValueError('Trainer used a different prepared partition')
    expected_order=list(selection['selected']);random.Random(protocol['training_candidate']['seed']).shuffle(expected_order)
    expected_schedule=[{'example_sha256':hashlib.sha256(r['id'].encode()).hexdigest(),'tokens':r['tokens'],'supervised_tokens':r['supervised_tokens']} for r in expected_order]
    if trainer_selection['schedule']!=expected_schedule or trainer_selection['excluded_records'] or trainer_selection['selected_records']!=selection['selected_records']:
        raise ValueError('Trainer changed frozen sample membership, tokens or order')
    for variant,expected in [('baseline',False),('adapted',True)]:
        health=read('artifact-'+variant+'.json')
        expected_hashes={k:v for k,v in training['adapter_files'].items() if k.endswith('.safetensors')} if expected else {}
        if health['profile_sha256']!=identity['input_hashes']['profile'] or health['adapter']!=expected or health['adapter_sha256']!=expected_hashes:
            raise ValueError('Evaluated model differs from trained artifact')
        if any(read('loading-'+variant+'.json').get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
            raise ValueError('Model loading had key mismatch')
    summary=assess(base,adapted,protocol,read('legacy-baseline.json'),read('legacy-adapted.json'),training)
    telemetry=list(csv.DictReader((root/'telemetry.csv').open()))
    epochs=[datetime.fromisoformat(r['utc']).timestamp() for r in telemetry]
    hw={'samples':len(telemetry),'errors':controller['telemetry']['errors'],'interval_seconds':5,
        'first_utc':telemetry[0]['utc'],'last_utc':telemetry[-1]['utc'],
        'max_sampling_gap_seconds':max(b-a for a,b in zip(epochs,epochs[1:])) if len(epochs)>1 else None}
    for column in ['temperature_c','power_w','utilization_percent','memory_used_mib']:
        values=[float(r[column]) for r in telemetry]
        hw[column]={'min':min(values),'median':statistics.median(values),'mean':statistics.mean(values),'max':max(values)}
    # Trapezoid estimate over observed samples, not utility-meter energy or whole-machine power.
    hw['sampled_gpu_energy_kwh']=sum((b-a)*(float(x['power_w'])+float(y['power_w']))/2 for a,b,x,y in zip(epochs,epochs[1:],telemetry,telemetry[1:]))/3_600_000
    summary['time']={'started_utc':controller['started_utc'],'finished_utc':controller['finished_utc'],
        'pipeline_seconds':controller['finished_epoch']-controller['started_epoch'],
        'training_update_seconds':training['training_update_seconds'],'stages':controller['stages']}
    summary['coverage']={k:selection[k] for k in ['input_records','eligible_records','selected_records','required_tasks','covered_tasks','kinds','input_tokens','supervised_tokens']}
    summary['hardware']=hw
    costs={}
    for name,value in [('base',base),('adapted',adapted)]:
        generations=[g for r in value['results'] for g in r['generations']]
        seconds=sum(g['seconds'] for g in generations);tokens=sum(g['generated_tokens'] for g in generations)
        costs[name]={'tasks':len(value['results']),'controller_seconds':sum(r['seconds'] for r in value['results']),
            'model_calls':sum(r['calls'] for r in value['results']),'successful_generation_seconds':seconds,
            'generated_tokens':tokens,'generated_tokens_per_second':tokens/seconds if seconds else None,
            'tool_seconds':sum(r.get('tool_seconds',0) for r in value['results']),
            'peak_generation_allocated_bytes':max((g['peak_allocated_bytes'] for g in generations),default=0)}
    summary['evaluation_cost']=costs
    write_json(root/'summary.json',summary)
    rows=['| Scope | Base | LoRA | Improved | Regressed |','| --- | --- | --- | --- | --- |']
    for label,key in [('New repository holdout','test'),('Historical regression','dev')]:
        v=summary['splits'][key];rows.append(f"| {label} | {v['baseline_passed']}/{v['pairs']} | {v['adapted_passed']}/{v['pairs']} | {len(v['improved'])} | {len(v['regressed'])} |")
    outcome='Passes the preregistered screen: advance only to multi-seed replication.' if summary['advance_to_replication'] else 'Does not pass the preregistered screen. Retain the unchanged base; do not promote this adapter.'
    (root/'README.md').write_text(f'''# 014 · Task-balanced 27B coding quality screen

**{outcome}**

{chr(10).join(rows)}

Legacy function regression: {summary['legacy']['base_passed']}/9 → {summary['legacy']['adapted_passed']}/9. Every pair and gate is in [summary.json](summary.json); the rule was fixed in [protocol.json](protocol.json) before any model rollout.

![Paired quality](comparison.png)

These are constructed defects in pinned real OSS repositories, not historical issues or SWE-bench scores. The two fresh repositories did not enter posttraining; base pretraining exposure is unknown. Tests are visible. The grader checks allowed source changes against a fresh snapshot; no additional hidden test suite is claimed. The 15 previously observed dev/test tasks are explicitly historical regression, not fresh evidence. One seed and two small repository clusters cannot establish broad coding or long-task quality.

## Training coverage and cost

The same 145 licensed source records were audited. Exactly {selection['selected_records']} records cover {selection['covered_tasks']}/{selection['required_tasks']} training tasks, including {selection['kinds'].get('verified-reference-repair',0)} verified reference repairs. Selection filters oversized actions and samples the full trajectories within task-balanced quotas. Task coverage alone does not mean every record is a patch; successful inspection/test actions remain in the dataset. No evaluation trace was added to training.

Frozen BF16 27B backbone, {training['trainable_parameters']:,} trainable LoRA parameters; {training['steps']} updates, {training['supervised_tokens']:,} supervised tokens, {training['training_update_seconds']:.1f} update seconds. Total screen wall time {summary['time']['pipeline_seconds']/3600:.3f} hours includes model loading, controls, evaluation, training and checkpoints. This is not a 27B full-parameter pretraining run. Peak training allocation {training['peak_allocated_bytes']/2**30:.2f} GiB; reserved {training['peak_reserved_bytes']/2**30:.2f} GiB. Saved adapter greedy probe matched before/after reload.

![Training diagnostics](training.png)

Loss is on different single examples at each step in one pass; it is not a heldout loss curve or evidence of improved coding quality. The comparison with 012 is observational: sample membership, action-length filtering and the deterministic 013 runtime differ, so gains cannot be attributed to balanced selection alone.

## Evaluation cost

Repository campaigns use the same greedy decoding, unmerged adapter serving path and per-task budgets. Base / candidate generated {costs['base']['generated_tokens']:,} / {costs['adapted']['generated_tokens']:,} tokens in {costs['base']['successful_generation_seconds']:.1f} / {costs['adapted']['successful_generation_seconds']:.1f} successful generation seconds, with {costs['base']['model_calls']} / {costs['adapted']['model_calls']} calls. These are sequential task measurements, not a production serving throughput benchmark. Tool time and full controller durations are separately retained in the summary; failed-request waits are not represented as successful generation time.

## Hardware and completion

Actual UTC: {controller['started_utc']} to {controller['finished_utc']}. {len(telemetry)} GPU samples, {hw['errors']} collection errors; peak {hw['temperature_c']['max']:.0f} °C and {hw['power_w']['max']:.1f} W. The timestamped record includes evaluation and loading as well as training. Sampled GPU energy estimate {hw['sampled_gpu_energy_kwh']:.3f} kWh excludes CPU, system power and unobserved intervals. [telemetry.csv](telemetry.csv) and per-stage durations retain the measurement scope.

Raw tasks, answers, trajectories, source snapshots, host information and weights stay private. Public artifacts contain code, frozen configuration, provenance IDs/hashes and numeric evidence. MIT applies to original project code; upstream licenses remain separate. No ModelScope weight release or daily-use quality certification is implied.
''')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8,4));labels=['Fresh holdout','Historical regression','Legacy functions'];groups=[summary['splits']['test'],summary['splits']['dev'],{'pairs':9,'baseline_passed':summary['legacy']['base_passed'],'adapted_passed':summary['legacy']['adapted_passed']}]
    for offset,key,label,color in [(-.18,'baseline_passed','Base','#65758b'),(.18,'adapted_passed','LoRA','#197d83')]:
        bars=ax.bar([i+offset for i in range(3)],[v[key]/v['pairs']*100 for v in groups],width=.35,label=label,color=color)
        for bar,v in zip(bars,groups):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{v[key]}/{v['pairs']}",ha='center',fontsize=9)
    ax.set(xticks=range(3),xticklabels=labels,ylabel='Tasks passed (%)',ylim=(0,112),title='One frozen candidate · constructed repository defects');ax.legend();fig.tight_layout();fig.savefig(root/'comparison.png',dpi=170);plt.close(fig)
    metrics=read('training/metrics.json');fig,axes=plt.subplots(1,2,figsize=(10,3.7));steps=[r['step'] for r in metrics]
    axes[0].plot(steps,[r['loss'] for r in metrics],color='#197d83',linewidth=1);axes[0].set(xlabel='Optimizer update',ylabel='Assistant-only cross entropy',title='Different example at each update')
    axes[1].plot(steps,[r['seconds'] for r in metrics],color='#65758b',linewidth=1);axes[1].set(xlabel='Optimizer update',ylabel='Seconds',title='Update time (excludes checkpoint writes)');fig.tight_layout();fig.savefig(root/'training.png',dpi=170);plt.close(fig)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args();result=render(args.run)
    print(json.dumps({'advance_to_replication':result['advance_to_replication'],'gate':result['gate']}))
