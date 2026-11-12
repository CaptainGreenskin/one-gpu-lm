"""Recompute 020 selection and release gates from complete public numeric evidence."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP.parents[1]))
sys.path.insert(0, str(EXP))
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.runstate import identity_digest
_spec = importlib.util.spec_from_file_location('curriculum_release_decision', EXP/'decision.py')
_decision = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_decision)
paired, select, release_gate = _decision.paired, _decision.select, _decision.release_gate


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_curricula(root, protocol):
    """Bind each dataset separately and check the declared model-visible contrast."""
    root = Path(root); frozen = protocol['frozen_files']; data = {}
    for name in ['repair', 'recovery']:
        path = root/('data-'+name)
        selected = json.loads((path/'selection.json').read_text())
        receipt = json.loads((path/'receipt.json').read_text())
        require(sha256(path/'selection.json') == frozen['runs/prepared-data020/'+name+'/selection.json']
                and sha256(path/'receipt.json') == selected['receipt_sha256']
                and selected['data_sha256'] == protocol['prepared_data_sha256'][name] == receipt['sha256'],
                'Prepared candidate data identity changed')
        rows = selected['selected']
        require(len(rows) == selected['selected_records'] == protocol['training_records'] == 330
                and len({r['id'] for r in rows}) == 330 and not selected['excluded']
                and selected['evaluation_traces_used'] is False and receipt['teacher_api_used'] is False,
                'Candidate data membership or provenance changed')
        require(selected['input_tokens'] == sum(r['tokens'] for r in rows)
                and selected['supervised_tokens'] == sum(r['supervised_tokens'] for r in rows) == protocol['supervised_tokens'][name]
                and selected['max_input_tokens'] == max(r['tokens'] for r in rows) <= 4096
                and selected['max_target_tokens'] == max(r['supervised_tokens'] for r in rows) <= 512,
                'Candidate token accounting or eligibility changed')
        data[name] = selected
    ids = [r['id'] for r in data['repair']['selected']]
    require(ids == [r['id'] for r in data['recovery']['selected']]
            and hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest() == protocol['semantic_order_sha256'],
            'Semantic task/slot order differs')
    shared = contrast = 0
    for a,b in zip(data['repair']['selected'], data['recovery']['selected']):
        if a['id'].endswith(':curriculum-late_repair'):
            require(a['record_sha256'] != b['record_sha256'], 'Recovery contrast is unchanged')
            contrast += 1
        else:
            require(a['record_sha256'] == b['record_sha256'], 'Common model-visible supervision differs')
            shared += 1
    require(shared == protocol['shared_identical_records'] == 275
            and contrast == protocol['contrasting_records'] == 55, 'Contrast membership differs')
    return data


def validate_campaign(campaign, ids, protocol, protocol_hash, adapter):
    identity = campaign['identity']
    rows = campaign['results']
    require(campaign['complete'] and identity['ids'] == ids and [r['id'] for r in rows] == ids,
            'Incomplete, duplicated or reordered campaign')
    require(identity['tasks_sha256'] == protocol['task_spec_sha256']
            and identity['admission_sha256'] == protocol['admission_sha256']
            and identity['protocol_sha256'] == protocol_hash and identity['variant'] == 'shared',
            'Campaign identity changed')
    require(all(protocol['frozen_files'].get(k) == v for k, v in identity['source_files'].items()),
            'Campaign executed unfrozen code')
    policy = protocol['variants']['shared']; budget = protocol['agent']
    for row in rows:
        generations = row['generations']; errors = row['backend_errors']
        require(type(row['passed']) is bool and row['agent_policy'] == policy and row['variant'] == 'shared',
                'Task policy or score schema changed')
        require(not row['passed'] or (row.get('returncode') == 0 and row.get('reason') == 'ok'),
                'A successful score requires a successful independent grader exit')
        require(row['calls'] == len(generations)+len(errors) <= budget['max_model_calls'], 'Wrong call accounting')
        require(row['generated_tokens'] == sum(g['generated_tokens'] for g in generations)
                and row['generated_tokens'] <= budget['max_total_generated_tokens'], 'Wrong output-token accounting')
        for g in generations:
            require(g['adapter'] is adapter and g['agent_policy'] == policy, 'Wrong evaluated variant')
            require(g['input_tokens'] <= budget['max_input_tokens']
                    and g['generated_tokens'] <= budget['max_generated_tokens_per_call'], 'Generation budget exceeded')
            require(g['context']['policy'] == 'full' and not g['context']['dropped_rounds'], 'History was truncated')
            require(type(g['action_format_valid']) is bool, 'Missing format outcome')
    generations = [g for row in rows for g in row['generations']]
    seconds = sum(g['seconds'] for g in generations)
    tokens = sum(g['generated_tokens'] for g in generations)
    return {'tasks': len(rows), 'passed': sum(r['passed'] for r in rows),
        'calls': sum(r['calls'] for r in rows), 'generation_calls': len(generations),
        'generated_tokens': tokens, 'processed_input_tokens': sum(g['input_tokens'] for g in generations),
        'generation_seconds': seconds, 'generated_tokens_per_second': tokens/seconds if seconds else None,
        'task_seconds': sum(r['seconds'] for r in rows), 'tool_seconds': sum(r['tool_seconds'] for r in rows),
        'format_invalid_calls': sum(not g['action_format_valid'] for g in generations),
        'output_limit_hits': sum(g['hit_generation_limit'] for g in generations),
        'backend_errors': dict(Counter(g.get('error', 'unknown') for r in rows for g in r['backend_errors'])),
        'exit_status': dict(Counter(r['exit_status'] for r in rows)),
        'failed_without_any_edit': sum(not r['passed'] and not any(s['changed_editable_files'] for s in r['source_progress']) for r in rows),
        'failed_after_edit': sum(not r['passed'] and any(s['changed_editable_files'] for s in r['source_progress']) for r in rows)}


def assess(root):
    root = Path(root); read = lambda name: json.loads((root/name).read_text())
    collection = read('collection.json')
    require(all(sha256(root/name) == digest for name,digest in collection['files'].items()),
            'Collected numeric evidence changed after transfer')
    protocol = read('protocol.json'); plan = read('preparation-plan.json'); controller = read('controller.json')
    protocol_hash = sha256(root/'protocol.json'); frozen = protocol['frozen_files']
    require(controller['complete'] and controller['stage'] == 'completed' and not controller.get('error_type'), 'Incomplete controller')
    require(controller['protocol_sha256'] == protocol_hash and controller['telemetry']['thread_stopped'], 'Controller identity or sampler mismatch')
    pub = read('preregistration.json'); launch = read('launch.json')
    require(pub['anonymous_files_verified'] and pub['ci_conclusion'] == 'success'
            and pub['protocol_sha256'] == protocol_hash and pub['commit'] == launch['source_commit']
            and datetime.fromisoformat(pub['verified_utc']) < datetime.fromisoformat(controller['started_utc']),
            'Public preregistration must precede training')
    require(sha256(root/'preparation-plan.json') == frozen['experiments/020-repair-curriculum/preparation-plan.json'], 'Changed locked rules')
    selection = read('selection.json'); inventory = read('task-inventory.json')
    data = validate_curricula(root, protocol)
    require(len(inventory) == 118 and len({t['id'] for t in inventory}) == 118, 'Changed task inventory')
    admitted = {r['id']: r for r in read('admission.json')['tasks']}
    require(sha256(root/'admission.json') == protocol['admission_sha256'], 'Changed admission')
    require(all(t['id'] in admitted and all(t[k] == admitted[t['id']][k] for k in ['repository','split','family']) for t in inventory), 'Inventory differs from admission')
    cohorts = {'dev': [t['id'] for t in inventory if t['split'] == 'dev'],
        'historical': [t['id'] for t in inventory if t.get('evaluation_role') == 'historical_final_regression_not_selection']}
    cohorts['fresh'] = [t['id'] for t in inventory if t['split'] == 'test' and t['id'] not in cohorts['historical']]
    require(tuple(map(len, cohorts.values())) == (15,24,24), 'Changed cohort counts')
    audit = read('execution-audit.json'); integrity = read('postrun-integrity.json')
    require(identity_digest(inventory) == integrity['public_task_inventory_identity_sha256'],
            'Public historical/fresh labels differ from the frozen private task manifest')
    require(audit['audit_source_sha256'] == frozen['experiments/020-repair-curriculum/audit.py']
            and audit['selection_recomputed_without_test_scores'], 'Missing exact execution audit')
    require(integrity['protocol_sha256'] == protocol_hash and all(integrity.get(k) is True for k in
        ['all_frozen_file_hashes_match','source_snapshot_contents_unchanged','dependency_contents_unchanged','framework_contents_unchanged']),
        'Post-run content integrity failed')
    require(integrity['base_model_verification']['content_verified']
            and integrity['base_model_verification']['manifest_sha256'] == frozen['experiments/011-coding-model-selection/weights-manifest.json']
            and all(integrity['base_model_verification'][k] == protocol[k] for k in ['model_id','revision'])
            and integrity['training_input_audit_sha256'] == frozen['training-input-audit.json']
            and sha256(root/'training-input-audit.json') == frozen['training-input-audit.json'],
            'Model or training-input rights audit identity changed')
    training = {}
    for name in ['repair', 'recovery']:
        selected_data = data[name]
        expected_order = list(selected_data['selected'])
        random.Random(plan['shared_recipe']['seed']).shuffle(expected_order)
        schedule = [{'example_sha256': hashlib.sha256(r['id'].encode()).hexdigest(), 'tokens': r['tokens'],
                     'supervised_tokens': r['supervised_tokens']} for r in expected_order]
        result = read('training-'+name+'/result.json')
        identity = read('training-'+name+'/identity.json'); chosen = read('training-'+name+'/selection.json')
        metrics = read('training-'+name+'/metrics.json')
        require(result['status'] == 'completed' and result['steps'] == len(metrics) == protocol['training_records'], 'Training is incomplete')
        require([r['step'] for r in metrics] == list(range(1,len(metrics)+1)), 'Noncontiguous updates')
        require(all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in metrics), 'Nonfinite training metrics')
        require(chosen['schedule'] == schedule and chosen['excluded_records'] == 0 and chosen['selected_records'] == len(schedule), 'Training membership or order changed')
        require(result['supervised_tokens'] == sum(r['supervised_tokens'] for r in metrics) == protocol['supervised_tokens'][name], 'Training token coverage changed')
        require(identity['input_hashes']['data'] == protocol['prepared_data_sha256'][name]
                and identity['input_hashes']['data_receipt'] == selected_data['receipt_sha256']
                and identity['input_hashes']['profile'] == frozen['experiments/020-repair-curriculum/profile-'+name+'.json'], 'Training input identity mismatch')
        require(all(frozen.get(k) == v for k,v in identity['source_files'].items()), 'Training code changed')
        require(all(result.get(k) is True for k in ['frozen_sample_unchanged','frozen_gradients_absent','adapter_changed','saved_reload_greedy_ids_equal']), 'Training update or reload check failed')
        require(audit['training'][name]['adapter_files'] == result['adapter_files'] and audit['training'][name]['checkpoint_verified'], 'Checkpoint audit mismatch')
        tensor_check = integrity['adapter_tensor_audits'][name]
        require(tensor_check['parameters'] == result['trainable_parameters'] and tensor_check['dtypes'] == ['F32']
                and tensor_check['file_sha256'] == result['adapter_files']['adapter_model.safetensors'], 'Adapter tensor audit differs from actual trained file')
        require(result['input_tokens'] == sum(r['tokens'] for r in metrics) == selected_data['input_tokens'],
                'Training input-token exposure changed')
        require([{k:r[k] for k in ['example_sha256','tokens','supervised_tokens']} for r in metrics] == schedule,
                'Actual updates differ from the frozen semantic schedule')
        training[name] = result
    # Semantic task/slot order is verified above; token lengths intentionally differ.
    arms = ['dev-base','dev-repair','dev-recovery'] + (['final-base','final-selected'] if selection['selected'] else [])
    campaigns = {arm: read('campaigns/'+arm+'.json') for arm in arms}; costs = {}
    for arm, campaign in campaigns.items():
        candidate = None if arm.endswith('-base') else selection['selected'] if arm == 'final-selected' else arm.removeprefix('dev-')
        ids = cohorts['dev'] if arm.startswith('dev-') else [t['id'] for t in inventory if t['split'] == 'test']
        costs[arm] = validate_campaign(campaign, ids, protocol, protocol_hash, candidate is not None)
        require(audit['campaign_sha256'][arm] == sha256(root/'campaigns'/(arm+'.json')), 'Audit belongs to a different campaign')
        rows = audit['variants'][arm]
        require([r['id'] for r in rows] == ids and all(r['progress_and_native_turns_verified'] and r['final_source_verified'] for r in rows)
                and sum(r['input_token_hashes_verified'] for r in rows) == costs[arm]['calls'], 'Incomplete execution audit')
        health = read('artifacts/'+arm+'.json'); loading = read('loading/'+arm+'.json')
        hashes = {k:v for k,v in training[candidate]['adapter_files'].items() if k.endswith('.safetensors')} if candidate else {}
        require(all(health[k] == protocol[k] for k in ['model_id','revision'])
                and health['profile_sha256'] == frozen['experiments/020-repair-curriculum/profile-repair.json']
                and health['adapter'] is (candidate is not None) and health['adapter_sha256'] == hashes, 'Wrong served model identity')
        require(not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']), 'Model loading mismatch')
    recomputed = select(campaigns['dev-base'], {n:campaigns['dev-'+n] for n in ['repair','recovery']}, cohorts['dev'], plan)
    require({k:v for k,v in selection.items() if k != 'final_test_opened'} == {k:v for k,v in recomputed.items() if k != 'final_test_opened'}, 'Candidate selected using a different rule')
    require(selection['final_test_opened'] == (selection['selected'] is not None), 'Test exposure mismatch')
    controls = read('loaded-controls.json')
    require(controls['complete'] and len(controls['clean']) == protocol['repository_count']
            and controls['task_spec_sha256'] == protocol['task_spec_sha256'] and all(r['passed'] for r in controls['clean']), 'GPU-loaded controls failed')
    comparisons = {}; legacy = {}; pairs = []
    within_budget = controller['finished_epoch'] <= controller['deadline_epoch'] and audit['original_wall_budget_respected']
    if selection['selected']:
        for name in ['historical','fresh']:
            subset = lambda c: dict(c, results=[r for r in c['results'] if r['id'] in cohorts[name]])
            comparisons[name] = paired(subset(campaigns['final-base']), subset(campaigns['final-selected']), cohorts[name])
        for arm, expected_adapter in [('final-base',False),('final-selected',True)]:
            value = read('legacy-'+arm+'.json'); rows = value['results']
            require(value['complete'] and value['task_sha256'] == frozen['private/legacy-functions.jsonl']
                    and len(rows) == len({r['id'] for r in rows}) == 9 and all(r['adapter'] is expected_adapter for r in rows), 'Legacy checks incomplete or unpaired')
            legacy_audit = integrity.get('legacy_input_audits', {}).get(arm, {})
            require(legacy_audit.get('result_sha256') == sha256(root/('legacy-'+arm+'.json'))
                    and legacy_audit.get('task_sha256') == value['task_sha256']
                    and legacy_audit.get('adapter') is expected_adapter
                    and [r['id'] for r in legacy_audit.get('results', [])] == [r['id'] for r in rows]
                    and all(a['input_verified'] is True and a['input_tokens_sha256'] == r['input_tokens_sha256']
                            for a, r in zip(legacy_audit['results'], rows)),
                    'Legacy result is not bound to the actual-input audit')
            legacy[arm] = {'passed':sum(r['passed'] for r in rows),'tasks':len(rows),'ids':[r['id'] for r in rows]}
        require(legacy['final-base']['ids'] == legacy['final-selected']['ids'], 'Legacy membership differs')
        export = integrity['export']
        execution_ok = within_budget and export.get('created') is True and export.get('adapter_files_match_training_and_evaluated_checkpoint') is True and export.get('candidate') == selection['selected']
        checks = {'legacy_passes':legacy['final-selected']['passed'], 'save_reload_greedy_match': True,
            'rights_and_secret_scan_pass': integrity['rights_and_secret_scan_pass'],
            'all_execution_audits_pass': execution_ok, 'single_training_seed_and_repository_cluster_limits_disclosed': True}
        decision = release_gate(comparisons['fresh'], comparisons['historical'], checks, plan)
    else:
        require(integrity.get('unselected_final_test_artifacts_absent') is True,
                'Missing remote proof that unselected final-test artifacts are absent')
        require(not any((root/'campaigns'/(name+'.json')).exists() for name in ['final-base','final-selected']), 'Unselected candidate opened final scores')
        decision = {'accepted':False,'reason':'development_gate_failed','failed_gates':['development_selection'],
                    'final_test_opened':False,'advanced_coding_or_daily_use_certified':False}
    for label, base_arm, candidate_arm in [('dev-repair','dev-base','dev-repair'),('dev-recovery','dev-base','dev-recovery')]+([('final','final-base','final-selected')] if selection['selected'] else []):
        base = {r['id']:r for r in campaigns[base_arm]['results']}
        for row in campaigns[candidate_arm]['results']:
            b = base[row['id']]
            pairs.append({'comparison':label,'id':row['id'],'repository':row['repository'],
                'cohort':'development' if label.startswith('dev') else 'historical' if row['id'] in cohorts['historical'] else 'fresh',
                'base_passed':b['passed'],'candidate_passed':row['passed'],
                'outcome':'improved' if row['passed'] and not b['passed'] else 'regressed' if b['passed'] and not row['passed'] else 'both_passed' if row['passed'] else 'both_failed',
                'base_calls':b['calls'],'candidate_calls':row['calls'], 'base_output_tokens':b['generated_tokens'],
                'candidate_output_tokens':row['generated_tokens'],'base_seconds':b['seconds'],'candidate_seconds':row['seconds'],
                **{label+'_'+key:value for label,r in [('base',b),('candidate',row)] for key,value in {
                    'exit_status':r['exit_status'], 'grade_reason':r.get('reason'), 'grade_returncode':r.get('returncode'),
                    'first_edit_call':next((s['call_index'] for s in r['source_progress'] if s['changed_editable_files']),None),
                    'final_changed_files':len(r['modified_source_files']),
                    'invalid_format_calls':sum(not g['action_format_valid'] for g in r['generations'])}.items()}})
    samples = []
    for file in sorted(root.glob('telemetry-attempt-*.csv')):
        with file.open() as stream:
            samples.extend(csv.DictReader(stream))
    epochs = [datetime.fromisoformat(r['utc']).timestamp() for r in samples]
    require(len(samples) >= 2 and all(b>a for a,b in zip(epochs,epochs[1:])), 'Invalid hardware coverage')
    hardware = {'samples':len(samples),'last_attempt_collection_errors':controller['telemetry']['errors'],
        'sampled_span_seconds':epochs[-1]-epochs[0],'max_gap_seconds':max(b-a for a,b in zip(epochs,epochs[1:])),
        'sampled_gpu_energy_kwh':sum((b-a)*(float(x['power_w'])+float(y['power_w']))/2 for a,b,x,y in zip(epochs,epochs[1:],samples,samples[1:]))/3600000}
    for key in ['temperature_c','memory_used_mib','power_w','utilization_percent']:
        values = [float(r[key]) for r in samples]
        hardware[key] = {'min':min(values),'median':statistics.median(values),'max':max(values)}
    return {'schema_version':1,'selection':selection,'comparisons':comparisons,'release_decision':decision,
        'training':training,'data_exposure':{n:{k:v[k] for k in ['input_tokens','supervised_tokens','max_input_tokens','max_target_tokens']} for n,v in data.items()},'legacy':legacy,'evaluation_cost':costs,'pairs':pairs,'hardware':hardware,
        'time':{'started_utc':controller['started_utc'],'finished_utc':controller['finished_utc'],
                'wall_seconds':controller['finished_epoch']-controller['started_epoch'],'within_original_budget':within_budget},
        'weights_public':False,'limitations':plan['limitations'],
        'report_sha256':sha256(Path(__file__))}


def render(root):
    root = Path(root); summary = assess(root)
    write_json(root/'summary.json', summary)
    write_json(root/'release-decision.json', summary['release_decision'])
    write_json(root/'changed-pair-behavior.json',{'scope':'Observed behavior of all changed pairs; not a causal explanation or a rerun',
        'cases':[r for r in summary['pairs'] if r['outcome'] in {'improved','regressed'}]})
    with (root/'paired-outcomes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary['pairs'][0]));writer.writeheader();writer.writerows(summary['pairs'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt':'one-gpu-lm-020','font.family':'DejaVu Sans'})
    colors={'repair':'#197d83','recovery':'#b45309'}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name in ['repair','recovery']:
        metrics=json.loads((root/('training-'+name)/'metrics.json').read_text());steps=[r['step'] for r in metrics]
        for ax,key in zip(axes,['loss','seconds']):ax.plot(steps,[r[key] for r in metrics],label=name,color=colors[name],alpha=.8)
    axes[0].set(title='Single-example loss; matched task/slot order',xlabel='Optimizer update',ylabel='Assistant-only cross entropy')
    axes[1].set(title='Update time (excludes checkpoints)',xlabel='Optimizer update',ylabel='Seconds')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    for ext in ['png','svg']:fig.savefig(root/('training-curves.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {})
    plt.close(fig)
    selection=summary['selection'];groups=[('Development', ['dev-base','dev-repair','dev-recovery'])]
    if selection['selected']:groups += [(name.title(),None) for name in ['historical','fresh']]
    fig,axes=plt.subplots(1,len(groups),figsize=(5*len(groups),4),squeeze=False,layout='constrained')
    for ax,(label,arms) in zip(axes[0],groups):
        if arms:
            values=[summary['evaluation_cost'][a]['passed'] for a in arms];labels=['base','repair','recovery'];total=15
        else:
            v=summary['comparisons'][label.lower()];values=[v['base_passed'],v['candidate_passed']];labels=['base',selection['selected']];total=v['pairs']
        ax.bar(labels,values,color=[{'base':'#64748b',**colors}[name] for name in labels])
        ax.set(title=label,xlabel='Model variant',ylabel='Independently graded passes',ylim=(0,total+1))
        for x,y in enumerate(values):ax.text(x,y+.1,f'{y}/{total}',ha='center')
    for ext in ['png','svg']:fig.savefig(root/('success-comparison.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {})
    plt.close(fig)
    accepted=summary['release_decision']['accepted']
    statement='Meets the fixed experimental adapter release gates; uploading still requires package publication and public verification.' if accepted else 'Does not meet the fixed release gates. Keep the base; do not publish candidate weights.'
    devtable='\n'.join(f"| {name} | {v['base_passed']}/15 | {v['candidate_passed']}/15 | {len(v['gained'])} | {len(v['regressed'])} |" for name,v in selection['comparisons'].items())
    finaltable='\n'.join(f"| {name} | {v['base_passed']}/24 | {v['candidate_passed']}/24 | {len(v['gained'])} | {len(v['regressed'])} |" for name,v in summary['comparisons'].items())
    trainlines='\n'.join(f"| {name} | {v['steps']} | {v['trainable_parameters']:,} | {v['input_tokens']:,} | {v['supervised_tokens']:,} | {v['training_update_seconds']:.1f} | {v['peak_allocated_bytes']/2**30:.2f} |" for name,v in summary['training'].items())
    costlines='\n'.join(f"| {name} | {v['passed']}/{v['tasks']} | {v['calls']} | {v['generated_tokens']} | {v['format_invalid_calls']} | {v['task_seconds']/60:.2f} |" for name,v in summary['evaluation_cost'].items())
    behaviorlines='\n'.join(f"| {name} | {v['failed_without_any_edit']} | {v['failed_after_edit']} |" for name,v in summary['evaluation_cost'].items())
    final_section=('Final scores remain unopened because neither candidate passed development selection.' if not selection['selected'] else
        '| Final cohort | Base | Selected candidate | Improved | Regressed |\n| --- | --- | --- | --- | --- |\n'+finaltable+'\n\nLegacy checks: '+str(summary['legacy']['final-base']['passed'])+'/9 base; '+str(summary['legacy']['final-selected']['passed'])+'/9 selected candidate.')
    hardware=summary['hardware'];timing=summary['time']
    body=f'''# 020 · Matched repair and failure-recovery curricula

**{statement}**

This is a small executed repair-trajectory pilot, not full-parameter pretraining or a general coding/long-task certification. The exact base, 55 training tasks, 330 records, one-pass task/slot order, shared learning rate, inference budgets and release gates were publicly frozen before weight updates. [Protocol](protocol.json), [preregistration](preregistration.json), [complete evidence and gates](summary.json).

| Candidate | Dev base | Candidate | Improved pairs | Regressed pairs |
| --- | --- | --- | --- | --- |
{devtable}

Selected using development only: **{selection['selected'] or 'none'}**. Selection requires net gain >=1 and at most one regression, then most passes, fewest generated tokens, lower learning rate, then declared candidate order. Both learning rates are 2e-6. Test scores never choose a candidate.

{final_section}

![Independent final-source success](success-comparison.png)

| Candidate | Updates | Trainable parameters | Input tokens | Supervised tokens | Update seconds | Peak allocation GiB |
| --- | --- | --- | --- | --- | --- | --- |
{trainlines}

Both candidates start from the identical pinned base and use the same 330 semantic task/slot IDs in seed-101 order. The 275 common records have identical model-visible token hashes; only 55 late-repair records differ. Three repair targets per task cover ordinary, stalled and late repair. Actual input and supervised token totals differ as reported above: this is not equal-token or equal-FLOP training. The base is frozen BF16; rank-8 adapters use FP32 and learning rate 2e-6 in both groups. Incorrect attempts are observed history only, never supervised targets. Both changed adapter weights, had no frozen-parameter gradients, passed a sampled frozen-weight integrity check, and matched greedy token IDs before/after adapter reload. This is not a full-backbone equality proof. The save/reload probe checks serialization, not task quality.

![Training diagnostics](training-curves.png)

Every update uses a different example. Targets and history can differ in the 55 contrast slots, so pointwise losses are not an identical-example comparison. These curves show training diagnostics, not heldout loss or generalization; a lower loss does not establish a better coding model.

| Evaluation arm | Passes | Model calls | Generated tokens | Invalid formats | Total task minutes |
| --- | --- | --- | --- | --- | --- |
{costlines}

| Evaluation arm | Failed with no observed allowed-source edit | Failed after an observed allowed-source edit |
| --- | --- | --- |
{behaviorlines}

Edit categories use the source snapshots after executed tools, not the model's claim that it made a repair. Only files allowed by the frozen task count; a change later reverted still counts as an observed edit. These counts describe the failure behavior and do not by themselves establish its cause.

Budgets are equal upper limits: 12 calls, 512 generated tokens/call, 6,144 generated tokens/task, 8,192 input tokens, and 900 seconds/task. Same native tool round trip, parser, full history and edit-progress policy. Actual usage differs and is retained. Candidate inference uses an unmerged adapter; sequential shared-host timings include that overhead and do not establish a production throughput comparison. [All paired outcomes](paired-outcomes.csv) retain regressions. Passing means the final independent grader accepts source changes in a fresh workspace, not simply an edit, an agent claim or a normal exit.

[Every improved and regressed pair](changed-pair-behavior.json) records the first observed edit, final changed-file count, format errors, call-limit/submit status and independent grader exit. A nonzero test exit labeled `runtime_error` is not automatically a GPU or controller failure. Conversely, a last-call repair can pass independent tests even when the agent then reaches its call limit. These are observed behavior differences, not a proven explanation of why training changed a result. There is one greedy rollout per task and arm; run-to-run and training-seed variability are not estimated.

For each executed legacy function check, [post-run integrity](postrun-integrity.json) binds the exact prompt-token hash, task order, model flag and private generated-text file hash to the public numeric result. This read-only audit was added during the run; it neither regenerates answers nor replays grading. Frozen training/evaluation inputs, selection rules and quality thresholds are unchanged. Audit/report source hashes identify the postprocessing version.

Actual UTC: {timing['started_utc']} to {timing['finished_utc']}; {timing['wall_seconds']/3600:.3f} pipeline hours. Original 12-hour budget respected: {timing['within_original_budget']}. {hardware['samples']} hardware samples, max gap {hardware['max_gap_seconds']:.2f}s, sampled peak {hardware['temperature_c']['max']:.0f}C / {hardware['power_w']['max']:.1f}W. GPU energy estimate {hardware['sampled_gpu_energy_kwh']:.3f}kWh uses trapezoids between samples; it excludes CPU/system power and can miss transients. Allocator memory peaks are separate from sampled board memory.

Limitations: deliberately constructed training failure controls, reused development tasks with prior feedback, constructed Python/JavaScript defects, visible upstream tests, unknown base pretraining overlap, one training seed and four small fresh-repository clusters. The development set has 14 original defects and one pre-freeze corrected no-op defect. No advanced multilingual coding, long-context or daily-use claim follows. Reviewed base/source licenses and known-pattern secret scans are bounded evidence, not a legal warranty. Raw tasks, commands, trajectories, corpora, machine identifiers, optimizer states and candidate weights are absent from this public results bundle. Original code is MIT; upstream licenses remain separate.
'''
    (root/'README.md').write_text(body)
    (root/'MODEL_CARD.md').write_text('# Experimental adapter decision record\n\n'+statement+'\n\n'
        'Base: `'+json.loads((root/'protocol.json').read_text())['model_id']+'` at `'+json.loads((root/'protocol.json').read_text())['revision']+'`.\n\n'
        'A LoRA adapter requires this exact base and is not a complete 27B model. No weights are included here. See [the measured outcome and limitations](README.md), [fixed rules](preparation-plan.json), [source provenance](sources.json) and [post-run integrity](postrun-integrity.json).\n\n'
        'Scope: experimental repository repair in isolated task sandboxes. One training seed; constructed Python/JavaScript defects; four fresh repository clusters only if final testing opened. No broad coding, long-task or production readiness certification. Training data contains no external teacher API responses or development/final answers. Failed training attempts are deliberately constructed, not sampled model mistakes. Preserve the base and source notices with any authorized adapter distribution.\n')
    write_json(root/'report-environment.json',{'python':platform.python_version(),'matplotlib':matplotlib.__version__,'report_sha256':sha256(Path(__file__))})
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();value=render(args.run)
    print(json.dumps({'selected':value['selection']['selected'],'release':value['release_decision']}))
