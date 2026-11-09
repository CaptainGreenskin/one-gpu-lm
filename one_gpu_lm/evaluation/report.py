"""Build paired repository results from an explicit public receipt directory."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics

from one_gpu_lm.common.receipts import sha256, write_json


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
    source, adapter, profile, data = read('source.json'), read('adapter.json'), read('profile.json'), read('data.json')
    for label,expected_adapter in [('base',False),('adapted',True)]:
        backend = source['backends'][label]
        if any(backend[key] != profile[key] for key in ['model_id','revision']) or backend['profile_sha256'] != sha256(run/'profile.json') or backend['adapter'] != expected_adapter:
            raise ValueError('Backend model/profile identity differs in '+label)
    if training['profile_sha256'] != sha256(run/'profile.json') or training['protocol_sha256'] != sha256(run/'protocol.json'):
        raise ValueError('Training profile or protocol identity differs')
    if training['data_sha256'] != data['sha256'] or data['source_campaign_sha256'] != sha256(run/'baseline.json') or data['task_spec_sha256'] != base['identity']['tasks_sha256']:
        raise ValueError('Training data provenance differs from the baseline campaign')
    if source['backends']['adapted']['adapter_sha256'] != adapter['files']:
        raise ValueError('Evaluated adapter differs from saved candidate')
    loading_names=['loading-base.json','loading-candidate.json','loading-training.json']
    for name in loading_names:
        loading=read(name)
        if any(loading.get(key) for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
            raise ValueError('Loading inspection failed in '+name)
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
    timing={}
    for name,value in [('base',base),('adapted',adapted)]:
        total=sum(r['seconds'] for r in value['results'])
        generation=sum(g.get('seconds',0) for r in value['results'] for g in r['generations'])
        tool=sum(r.get('tool_seconds',0) for r in value['results'])
        timing[name]={'controller_through_grading_seconds':total,'generation_seconds':generation,
            'agent_tool_seconds':tool,'grading_and_controller_overhead_seconds':total-generation-tool,
            'residual_includes':'fresh grading, workspace hashing, controller/RPC overhead, recovery pause and any unrecorded failed request wait; not a pure grading measurement'}
    summary['timing']=timing
    selection = read('selection.json')
    selected_tasks = Counter(r['id'].rsplit(':',1)[0] for r in selection['selected'])
    train_ids = {r['id'] for r in base['results'] if r['split']=='train'}
    if not set(selected_tasks).issubset(train_ids):
        raise ValueError('Selected supervision includes an unknown/non-training task')
    summary['supervision_coverage'] = {'selected_task_count':len(selected_tasks),
        'admitted_training_task_count':len(train_ids),'records_by_task':dict(sorted(selected_tasks.items())),
        'note':'A selected record can be an inspection or test command, not only a repair; task success and command return code were checked.'}
    write_json(run/'summary.json',summary)
    hw = hardware(run/'telemetry.csv',read('runtime.json')['telemetry_utc_offset_minutes'])
    write_json(run/'hardware.json',hw)
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
- Selected supervision: {len(selection['selected'])} records covering {len(selected_tasks)}/{len(train_ids)} admitted training tasks; {len(selection['excluded'])} excluded by frozen size/example budget. Kinds: `{json.dumps(selection['kinds'],sort_keys=True)}`. Records include inspection and test commands, not only code repairs; record count is not the number of distinct solved tasks.
- Recovery at completed steps: `{json.dumps(recovery,sort_keys=True)}`. Independent deterministic continuous/restart contracts are recorded in the preflight receipts. This does not cover arbitrary mid-command crashes or context compression.
- Sampling: {hw['samples']} GPU readings every nominal 5 s; maximum gap {hw['maximum_gap_seconds']:.2f} s; {hw['gaps_over_10_seconds']} gaps above 10 s. Observed peak {hw['temperature_c']['maximum']:.0f} °C, {hw['memory_used_mib']['maximum']:.0f} MiB used, {hw['power_w']['maximum']:.1f} W. Monitoring includes setup and the invalidated interface probe; it is not a training-only utilization benchmark.

![Task success and task time](comparison.png)

Task time covers controller initialization through fresh grading; it excludes initial snapshot preparation and model loading. Generation and agent-tool durations are measured separately; the residual combines fresh grading with controller/RPC/checkpoint overhead and recovery pauses, because this run did not retain a separate final-grading wall timer. Model calls, generated tokens, grading outcomes and exit status are separate fields. A valid patch may pass after the call budget is exhausted; successful self-reported tests alone never count. Cold compilation and filesystem/kernel cache effects are included where incurred and were not independently controlled. Commands run sequentially on one GPU backend; these are not isolated serving TTFT, continuous-batching throughput or concurrency benchmarks.

![Mixed-task training loss](loss.png)

This is assistant-token loss on different training contexts, **not a validation learning curve**. Frozen sample unchanged: `{training['frozen_sample_unchanged']}`; frozen gradients absent: `{training['frozen_gradients_absent']}`; adapter changed: `{training['adapter_changed']}`; save/reload exact greedy IDs: `{training['saved_reload_greedy_ids_equal']}`. The frozen sample check is not a whole-model hash.

## Scope, changes and reuse

Forty task controls admitted 36 tasks before formal model evaluation. Rejected tasks and narrow test-scope exclusions are disclosed in the protocol and admission receipts. `baseline-001` was stopped as an integration probe after only training-partition outputs: native Qwen XML tool syntax and a shared-UID thread limit required adapter/runtime fixes. Both formal campaigns use the same corrected code and budgets. The invalidated probe supplied no training examples or quality scores.

Source identities and file notices remain separate from the project's MIT orchestration license. No raw source corpus, task answers, trajectories, generated code or adapter weights are published. This report is not a ModelScope release or a legal clearance of upstream pretraining data. Private artifacts retain their applicable source notices.

Hashes alone do not make the private task corpus independently reproducible. Other users can run their own authorized tasks with [repo_lab](../../../../repo_lab/) and compare the same receipt formats. A future base must rerun this suite and new holdouts; changing a model ID alone does not validate loader, template, action parser, adapter target layers or inference performance.

Rebuild: `python -m repo_lab.report --run THIS_DIRECTORY`. Inputs and generated artifacts are allowlisted in `artifacts.json`.
"""
    (run/'README.md').write_text(report)
    if (run/'action-audit.json').exists():
        audit=read('action-audit.json')
        with (run/'README.md').open('a') as stream:
            stream.write('\n## Action-interface audit\n\n')
            for name in ['base','adapted']:
                stream.write(f"{name}: `{json.dumps(audit[name]['format_categories'],sort_keys=True)}`.\n\n")
            stream.write('These are heuristic interface categories. They distinguish missing actions, case/parameter mismatch and generation truncation; they do not assign all failures to the underlying coding model. See [action-audit.json](action-audit.json).\n')
    storage=read('adapter-storage.json')
    (run/'MODEL_CARD.md').write_text(f"""# Model card — private repository-agent LoRA candidate

Status: experimental local adapter. Daily-use acceptance: **false**. Small screening gate passed: **{summary['candidate_passes_this_screen']}**. No ModelScope weight release is represented by this card.

Base: [{profile['model_id']}](https://huggingface.co/{profile['model_id']}/tree/{profile['revision']}), revision `{profile['revision']}`. Loader `{profile['loader']}`, base dtype `{profile['dtype']}`; tokenizer/template and exact LoRA targets are fixed in `profile.json`. Rank {profile['lora']['r']}, alpha {profile['lora']['lora_alpha']}, {training['trainable_parameters']:,} trainable parameters. Saved adapter storage: `{json.dumps(storage,sort_keys=True)}`. The backbone dtype does not imply the adapter tensors use that same dtype; see `adapter-storage.json` for the actual safetensors header counts. This adapter requires the matching base; it is not a standalone model. Inference uses an unmerged PEFT adapter, so task timings include its extra execution overhead.

Intended use: a bounded engineering experiment in repository repair. Evaluation uses one greedy run per task and one training seed, visible tests and constructed defects. Final repository-disjoint tasks: {test['baseline_passed']}/{test['pairs']} base, {test['adapted_passed']}/{test['pairs']} candidate, {len(test['regressed'])} regressions. Previous simple fixtures: {old_adapted['passed']}/9 after training. These results do not establish advanced coding, autonomous long-task reliability, security, general language coverage or production serving performance.

The exact adapter file hashes are in `adapter.json` and match the evaluated backend identity in `source.json`. Saving/reloading reproduced a short greedy probe. Original orchestration code is MIT; the base uses its recorded {profile['model_license']} license and corpus components retain their own notices. This is not an audit or warranty of the base's upstream pretraining corpus.

Before a later ModelScope release: pass substantially broader independent tasks, review the precise data and weight distribution package, retain applicable licenses/notices, specify base/tokenizer/adapter hashes and dependency versions, and link a matching GitHub result. A future model version must rerun these checks; old LoRA weights are not assumed portable to a new architecture.
""")
    (run/'DATA_CARD.md').write_text(f"""# Data card — repository repair supervision and evaluation

Raw source, private task specifications, trajectories, reference patches and model outputs are not redistributed. Exact task-spec SHA-256: `{data['task_spec_sha256']}`. Exact training JSONL SHA-256: `{data['sha256']}`. Source identities and selected file notices are in the experiment-level source records.

Training examples: {data['records']} prepared records; {len(selection['selected'])} selected after the predeclared context/example limits; {len(selection['excluded'])} excluded. Selected kinds: `{json.dumps(selection['kinds'],sort_keys=True)}`. Repositories: `{json.dumps(selection['repositories'])}`. The selected records cover {len(selected_tasks)}/{len(train_ids)} admitted training tasks; counts per task are in `summary.json`. Successful baseline trajectories are self-generated local supervision, not outputs purchased from a teacher API. They include successful inspection/test commands and repairs from tasks that passed independent grading; each record is not necessarily a new repair demonstration. Failed training tasks may contribute separately labeled, executed reference repairs from their authorized snapshots, but the selected kind counts determine whether any actually entered training. Only the current assistant response contributes to loss.

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
    if (run/'action-audit.json').exists():
        allowed.append('action-audit.json')
    allowed.extend(loading_names)
    allowed.append('adapter-storage.json')
    write_json(run/'artifacts.json',{'schema_version':1,'weights_public':False,'raw_trajectories_public':False,
        'generator_sha256':sha256(__file__),'matplotlib_version':matplotlib.__version__,
        'files':{name:sha256(run/name) for name in allowed}})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    render(parser.parse_args().run)


if __name__ == "__main__":
    main()
