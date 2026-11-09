"""Build paired repository results from an explicit public receipt directory."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics

from training_lab.records import sha256, write_json


def paired(base, adapted, counts):
    if not base['complete'] or not adapted['complete']:
        raise ValueError('Incomplete campaign cannot enter paired analysis')
    if base['identity'] != adapted['identity']:
        raise ValueError('Task, protocol or execution code identities differ')
    expected = base['identity']['ids']
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Empty or duplicate expected IDs')
    for run in [base,adapted]:
        if [r['id'] for r in run['results']] != expected:
            raise ValueError('Missing, reordered or duplicate task result')
        if dict(Counter(r['split'] for r in run['results'])) != counts:
            raise ValueError('Split count differs from frozen admission')
    for expected_adapter, run in [(False,base),(True,adapted)]:
        if any(g['adapter'] != expected_adapter for r in run['results'] for g in r['generations']):
            raise ValueError('Generation came from the wrong model variant')
    rows = []
    for b,a in zip(base['results'],adapted['results']):
        if any(b[k] != a[k] for k in ['split','repository','family']):
            raise ValueError('Task metadata differs within pair')
        rows.append({'id':b['id'],'split':b['split'],'repository':b['repository'],
            'baseline_passed':b['passed'],'adapted_passed':a['passed'],
            'baseline_seconds':b['seconds'],'adapted_seconds':a['seconds'],
            'baseline_calls':b['calls'],'adapted_calls':a['calls'],
            'baseline_tokens':b['generated_tokens'],'adapted_tokens':a['generated_tokens'],
            'baseline_exit':b['exit_status'],'adapted_exit':a['exit_status'],
            'baseline_changed_files':len(b['modified_source_files']),
            'adapted_changed_files':len(a['modified_source_files'])})
    groups = {}
    for split in ['train','dev','test']:
        group = [r for r in rows if r['split']==split]
        groups[split] = {'pairs':len(group),'baseline_passed':sum(r['baseline_passed'] for r in group),
            'adapted_passed':sum(r['adapted_passed'] for r in group),
            'improved':[r['id'] for r in group if not r['baseline_passed'] and r['adapted_passed']],
            'regressed':[r['id'] for r in group if r['baseline_passed'] and not r['adapted_passed']],
            'baseline_seconds':sum(r['baseline_seconds'] for r in group),
            'adapted_seconds':sum(r['adapted_seconds'] for r in group)}
    return {'pairs':rows,'splits':groups}


def hardware(path, utc_offset_minutes):
    rows = list(csv.DictReader(path.open()))
    def column(prefix):
        return next(k for k in rows[0] if k.strip().startswith(prefix))
    times = [datetime.strptime(r[column('timestamp')].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(
        tzinfo=timezone(timedelta(minutes=utc_offset_minutes))) for r in rows]
    gaps = [(b-a).total_seconds() for a,b in zip(times,times[1:])]
    value = {'samples':len(rows),'first_utc':times[0].astimezone(timezone.utc).isoformat(),
        'last_utc':times[-1].astimezone(timezone.utc).isoformat(),'nominal_interval_seconds':5,
        'maximum_gap_seconds':max(gaps,default=0),'gaps_over_10_seconds':sum(g>10 for g in gaps),
        'scope':'entire experiment monitoring including setup and invalidated interface probe, not only training'}
    for prefix,name in [('temperature.gpu','temperature_c'),('power.draw','power_w'),('utilization.gpu','gpu_utilization_percent'),('memory.used','memory_used_mib')]:
        numbers = [float(r[column(prefix)].strip()) for r in rows]
        value[name] = {'minimum':min(numbers),'median':statistics.median(numbers),'mean':statistics.mean(numbers),'maximum':max(numbers)}
    return value


def render(run):
    read = lambda name: json.loads((run/name).read_text())
    base, adapted, protocol, training = read('baseline.json'),read('adapted.json'),read('protocol.json'),read('training.json')
    summary = paired(base,adapted,protocol['admitted_task_counts'])
    old_base, old_adapted = read('regression-base.json'),read('regression-adapted.json')
    if not old_base['complete'] or not old_adapted['complete'] or old_base['task_sha256'] != old_adapted['task_sha256']:
        raise ValueError('Legacy regression not paired')
    if [r['id'] for r in old_base['results']] != [r['id'] for r in old_adapted['results']] or len(old_adapted['results']) != 9:
        raise ValueError('Legacy regression membership mismatch')
    if training['status'] != 'completed':
        raise ValueError('Candidate training incomplete')
    source, adapter = read('source.json'), read('adapter.json')
    if source['backends']['adapted']['adapter_sha256'] != adapter['files']:
        raise ValueError('Evaluated adapter differs from saved candidate')
    test = summary['splits']['test']
    recovery = {name:[{'id':r['id'],'passed':r['passed'],'calls':r['calls'],'generated_tokens':r['generated_tokens']}
                     for r in value['results'] if r['recovered']] for name,value in [('base',base),('adapted',adapted)]}
    recovery_ok = all(recovery[name] and all(r['passed'] for r in recovery[name]) for name in recovery)
    gate = {'test_net_gain':test['adapted_passed']-test['baseline_passed'] >= protocol['promotion']['minimum_net_test_task_gain'],
            'no_test_regressions':len(test['regressed']) <= protocol['promotion']['maximum_regressed_test_tasks'],
            'legacy_9_all_pass':old_adapted['passed']==9,'real_task_recovery_passed':recovery_ok,
            'adapter_reload_exact':training['saved_reload_greedy_ids_equal']}
    summary.update(schema_version=1,gate=gate,candidate_passes_this_screen=all(gate.values()),daily_model_accepted=False,
        legacy={'base_passed':old_base['passed'],'adapted_passed':old_adapted['passed'],'tasks':9},recovery=recovery,
        inference_exit_counts={name:dict(Counter(r['exit_status'] for r in value['results'])) for name,value in [('base',base),('adapted',adapted)]})
    write_json(run/'summary.json',summary)
    hw = hardware(run/'telemetry.csv',read('runtime.json')['telemetry_utc_offset_minutes'])
    write_json(run/'hardware.json',hw)
    selection = read('selection.json')
    tables = ['| Partition | Base | LoRA | Improved | Regressed |','| --- | --- | --- | --- | --- |']
    for split,v in summary['splits'].items():
        tables.append(f"| {split} | {v['baseline_passed']}/{v['pairs']} | {v['adapted_passed']}/{v['pairs']} | {len(v['improved'])} | {len(v['regressed'])} |")
    outcome = 'Passes this small screening gate; requires new independent tasks before daily-use acceptance.' if summary['candidate_passes_this_screen'] else 'Does not pass the predeclared screening gate; retain the unmodified base as the candidate reference.'
    report = f"""# 012 · Repository agent / controlled posttraining

{outcome}

These are **constructed defects in real repositories**, not historical issue results, SWE-bench scores or evidence of advanced general coding ability. Training/dev/test repositories and task families are disjoint for this experiment; upstream exposure during the base model's pretraining is unknown. Tests are visible, and the grader reruns the declared upstream test scope from a fresh copy. There is no additional secret test suite.

{chr(10).join(tables)}

Legacy function fixtures: {old_base['passed']}/9 → {old_adapted['passed']}/9. Gate details and every paired task are in [summary.json](summary.json). Train scores describe the supervision source, not generalization. Dev results are diagnostic; test results did not select examples, hyperparameters or a checkpoint. One seed and only {test['pairs']} final tasks across two repositories cannot establish a broad capability gain.

## Actual execution

- Baseline UTC: {base['started_utc']} — {base['finished_utc']}.
- Candidate campaign UTC: {adapted['started_utc']} — {adapted['finished_utc']}.
- One LoRA pass: {training['steps']} updates, {training['training_seconds']:.1f} update seconds, {training['supervised_tokens']:,} supervised tokens; peak allocated {training['peak_allocated_bytes']/2**30:.2f} GiB.
- Selected supervision: {len(selection['selected'])} records; {len(selection['excluded'])} excluded by frozen size/example budget. Kinds: `{json.dumps(selection['kinds'],sort_keys=True)}`.
- Recovery at completed steps: `{json.dumps(recovery,sort_keys=True)}`. Independent deterministic continuous/restart contracts are recorded in the preflight receipts. This does not cover arbitrary mid-command crashes or context compression.
- Sampling: {hw['samples']} GPU readings every nominal 5 s; maximum gap {hw['maximum_gap_seconds']:.2f} s; {hw['gaps_over_10_seconds']} gaps above 10 s. Observed peak {hw['temperature_c']['maximum']:.0f} °C, {hw['memory_used_mib']['maximum']:.0f} MiB used, {hw['power_w']['maximum']:.1f} W. Monitoring includes setup and the invalidated interface probe; it is not a training-only utilization benchmark.

![Task success and task time](comparison.png)

Task time covers controller initialization through fresh grading; it excludes snapshot preparation and model loading. Model calls, generated tokens, final grading and exit status are separate fields. A valid patch may pass after the call budget is exhausted; successful self-reported tests alone never count. Commands run sequentially on one GPU backend; these are not isolated serving TTFT, continuous-batching throughput or concurrency benchmarks.

![Mixed-task training loss](loss.png)

This is assistant-token loss on different training contexts, **not a validation learning curve**. Frozen sample unchanged: `{training['frozen_sample_unchanged']}`; frozen gradients absent: `{training['frozen_gradients_absent']}`; adapter changed: `{training['adapter_changed']}`; save/reload exact greedy IDs: `{training['saved_reload_greedy_ids_equal']}`. The frozen sample check is not a whole-model hash.

## Scope, changes and reuse

Forty task controls admitted 36 tasks before formal model evaluation. Rejected tasks and narrow test-scope exclusions are disclosed in the protocol and admission receipts. `baseline-001` was stopped as an integration probe after only training-partition outputs: native Qwen XML tool syntax and a shared-UID thread limit required adapter/runtime fixes. Both formal campaigns use the same corrected code and budgets. The invalidated probe supplied no training examples or quality scores.

Source identities and file notices remain separate from the project's MIT orchestration license. No raw source corpus, task answers, trajectories, generated code or adapter weights are published. This report is not a ModelScope release or a legal clearance of upstream pretraining data. Private artifacts retain their applicable source notices.

Hashes alone do not make the private task corpus independently reproducible. Other users can run their own authorized tasks with [repo_lab](../../../../repo_lab/) and compare the same receipt formats. A future base must rerun this suite and new holdouts; changing a model ID alone does not validate loader, template, action parser, adapter target layers or inference performance.

Rebuild: `python -m repo_lab.report --run THIS_DIRECTORY`. Inputs and generated artifacts are allowlisted in `artifacts.json`.
"""
    (run/'README.md').write_text(report)
    profile=read('profile.json');data=read('data.json')
    (run/'MODEL_CARD.md').write_text(f"""# Model card — private repository-agent LoRA candidate

Status: experimental local adapter. Daily-use acceptance: **false**. Small screening gate passed: **{summary['candidate_passes_this_screen']}**. No ModelScope weight release is represented by this card.

Base: [{profile['model_id']}](https://huggingface.co/{profile['model_id']}/tree/{profile['revision']}), revision `{profile['revision']}`. Loader `{profile['loader']}`, dtype `{profile['dtype']}`; tokenizer/template and exact LoRA targets are fixed in `profile.json`. Rank {profile['lora']['r']}, alpha {profile['lora']['lora_alpha']}, {training['trainable_parameters']:,} trainable parameters. This adapter requires the matching base; it is not a standalone model.

Intended use: a bounded engineering experiment in repository repair. Evaluation uses one greedy run per task and one training seed, visible tests and constructed defects. Final repository-disjoint tasks: {test['baseline_passed']}/{test['pairs']} base, {test['adapted_passed']}/{test['pairs']} candidate, {len(test['regressed'])} regressions. Previous simple fixtures: {old_adapted['passed']}/9 after training. These results do not establish advanced coding, autonomous long-task reliability, security, general language coverage or production serving performance.

The exact adapter file hashes are in `adapter.json` and match the evaluated backend identity in `source.json`. Saving/reloading reproduced a short greedy probe. Original orchestration code is MIT; the base uses its recorded {profile['model_license']} license and corpus components retain their own notices. This is not an audit or warranty of the base's upstream pretraining corpus.

Before a later ModelScope release: pass substantially broader independent tasks, review the precise data and weight distribution package, retain applicable licenses/notices, specify base/tokenizer/adapter hashes and dependency versions, and link a matching GitHub result. A future model version must rerun these checks; old LoRA weights are not assumed portable to a new architecture.
""")
    (run/'DATA_CARD.md').write_text(f"""# Data card — repository repair supervision and evaluation

Raw source, private task specifications, trajectories, reference patches and model outputs are not redistributed. Exact task-spec SHA-256: `{data['task_spec_sha256']}`. Exact training JSONL SHA-256: `{data['sha256']}`. Source identities and selected file notices are in the experiment-level source records.

Training examples: {data['records']} prepared records; {len(selection['selected'])} selected after the predeclared context/example limits; {len(selection['excluded'])} excluded. Selected kinds: `{json.dumps(selection['kinds'],sort_keys=True)}`. Repositories: `{json.dumps(selection['repositories'])}`. Successful baseline trajectories are self-generated local supervision, not outputs purchased from a teacher API. Failed training tasks may contribute separately labeled, executed reference repairs from their authorized snapshots. Only the current assistant response contributes to loss.

Task controls admitted {sum(protocol['admitted_task_counts'].values())} of {sum(protocol['target_task_counts'].values())} constructed tasks. Frozen counts: `{json.dumps(protocol['admitted_task_counts'],sort_keys=True)}`. Training, development and final tasks use different repositories and API families; no development/test answers or trajectories enter SFT. This separation does not prove absence from base-model pretraining, nor does it make a {test['pairs']}-task final sample representative.

Public upstream tests are visible. Independent final grading reconstructs a fresh task and submits allowed source files only; it does not introduce new hidden assertions. Rejected tasks and excluded timing/thread-stress tests remain disclosed. The invalidated interface probe supplies no supervision.

The provenance records preserve repository revisions, file/license hashes and required notices. The project MIT license does not replace third-party source terms. Hashes cannot recreate withheld content: contributors can use the public input contract to evaluate their own authorized repositories, while exact reproduction of this private dataset needs the corresponding source preparation and task specifications.
""")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    splits=['train','dev','test']; x=list(range(3))
    axes[0].bar([i-.18 for i in x],[summary['splits'][s]['baseline_passed'] for s in splits],width=.36,label='Base')
    axes[0].bar([i+.18 for i in x],[summary['splits'][s]['adapted_passed'] for s in splits],width=.36,label='LoRA')
    axes[0].set(xticks=x,xticklabels=[f"{s} (n={summary['splits'][s]['pairs']})" for s in splits],ylabel='Tasks passed',title='Paired constructed repository tasks')
    axes[0].legend()
    maximum=max(max(r['baseline_seconds'],r['adapted_seconds']) for r in summary['pairs'])
    for split in splits:
        group=[r for r in summary['pairs'] if r['split']==split]
        axes[1].scatter([r['baseline_seconds'] for r in group],[r['adapted_seconds'] for r in group],label=split,alpha=.8)
    axes[1].plot([0,maximum],[0,maximum],'k--',alpha=.35)
    axes[1].set(xlabel='Base task seconds',ylabel='LoRA task seconds',title='Includes tools and independent grading')
    axes[1].legend()
    fig.tight_layout();fig.savefig(run/'comparison.png',dpi=170);plt.close(fig)
    curve=read('curve.json');fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([r['step'] for r in curve],[r['loss'] for r in curve],'o-',markersize=3)
    ax.set(xlabel='Optimizer update',ylabel='Assistant-token loss',title='One pass: different task contexts at each step')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(run/'loss.png',dpi=170);plt.close(fig)
    allowed=['README.md','MODEL_CARD.md','DATA_CARD.md','summary.json','hardware.json','comparison.png','loss.png','baseline.json','adapted.json',
        'protocol.json','profile.json','training.json','selection.json','data.json','curve.json','adapter.json',
        'regression-base.json','regression-adapted.json','telemetry.csv','runtime.json','pipeline.json','source.json']
    write_json(run/'artifacts.json',{'schema_version':1,'weights_public':False,'raw_trajectories_public':False,
        'generator_sha256':sha256(__file__),'matplotlib_version':matplotlib.__version__,
        'files':{name:sha256(run/name) for name in allowed}})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    render(parser.parse_args().run)
