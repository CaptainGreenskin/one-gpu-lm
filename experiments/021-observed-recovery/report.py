"""Recompute 021 comparisons from audited public numbers, without opening raw tasks."""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys

EXP=Path(__file__).resolve().parent
sys.path.insert(0,str(EXP.parents[1]))
from one_gpu_lm.common.receipts import sha256,write_json
from one_gpu_lm.common.runstate import identity_digest

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value);return value

decision=module(EXP/'decision.py','observed_decision')
prior=module(EXP.parent/'020-repair-curriculum/report.py','parent_report020')

def require(value,message):
    if not value:raise ValueError(message)

def assess(root):
    root=Path(root);read=lambda name:json.loads((root/name).read_text())
    collected=read('collection.json')
    require(all(sha256(root/n)==h for n,h in collected['files'].items()),'Collected evidence changed')
    protocol=read('protocol.json');plan=read('preparation-plan.json');controller=read('controller.json')
    require(controller['complete'] and controller['stage']=='completed' and not controller.get('error_type'),'Incomplete controller')
    require(controller['telemetry']['thread_stopped'] and controller['protocol_sha256']==sha256(root/'protocol.json'),'Controller/sampler identity differs')
    require(sha256(root/'preparation-plan.json')==protocol['frozen_files']['experiments/021-observed-recovery/preparation-plan.json'],'Changed fixed rules')
    pub=read('preregistration.json');launch=read('launch.json')
    require(pub['anonymous_files_verified'] and pub['ci_conclusion']=='success' and pub['protocol_sha256']==sha256(root/'protocol.json')
            and pub['commit']==launch['source_commit'] and datetime.fromisoformat(pub['verified_utc'])<datetime.fromisoformat(controller['started_utc']),
            'Public preregistration must precede collection')
    inventory=read('task-inventory.json');integrity=read('postrun-integrity.json')
    require(len(inventory)==len({t['id'] for t in inventory})==142,'Task inventory is incomplete')
    require(identity_digest(inventory)==integrity['public_task_inventory_identity_sha256'],'Cohort metadata differs from frozen private tasks')
    require(integrity['protocol_sha256']==sha256(root/'protocol.json') and all(integrity.get(k) is True for k in
            ['all_frozen_file_hashes_match','source_snapshot_contents_unchanged','dependency_contents_unchanged','framework_contents_unchanged']),
            'Source/runtime integrity audit failed')
    require(integrity['base_model_verification']['content_verified'] and all(integrity['base_model_verification'][k]==protocol[k]
            for k in ['model_id','revision']),'Wrong base model')
    require(sha256(root/'admission.json')==protocol['admission_sha256'],'Admission changed')
    controls=read('loaded-controls.json')
    require(controls['complete'] and len(controls['clean'])==protocol['repository_count'] and all(r['passed'] for r in controls['clean']),
            'Model-loaded clean controls failed')
    data=read('derived-data.json');observed=read('observed-input-audit.json');inputs=read('training-input-audit.json')
    require(observed['audit_source_sha256']==sha256(EXP/'audit_observed.py') and inputs['audit_source_sha256']==sha256(EXP/'audit_inputs.py')
            and integrity['script_sha256']==sha256(EXP/'postrun_integrity.py'),'Audit source version differs')
    require(observed['passed'] and observed['derived_data_sha256']==sha256(root/'derived-data.json')
            and observed['prepared_data_sha256']==data['prepared_data_sha256'] and observed['whole_context_and_masking_verified'],
            'Derived data lacks exact input evidence')
    require(inputs['passed'] and inputs['data_sha256']=={'observed':data['prepared_data_sha256']}
            and integrity['training_input_audit_sha256']==sha256(root/'training-input-audit.json'),'Actual training-input screen failed')
    recovery=read('derivation-recovery.json'); recovered=observed['derivation_recovery']
    require(recovered['passed'] and recovered['audit_source_sha256']==sha256(EXP/'audit_recovery.py')
            and recovered['recovery_receipt_sha256']==sha256(root/'derivation-recovery.json')
            and recovery['builder_sha256']==sha256(EXP/'build_data_recovery.py')
            and recovery['derived_data_sha256']==sha256(root/'derived-data.json'), 'Recovery evidence differs')
    require(recovery['original_started_epoch']==controller['started_epoch']
            and recovery['original_deadline_epoch']==controller['deadline_epoch']
            and not recovery['model_collection_rerun'] and not recovery['deadline_extended'], 'Recovery changed the budget or collection')
    selected_data=read('data-observed/selection.json');receipt=read('data-observed/receipt.json')
    require(sha256(root/'data-observed/selection.json')==data['selection_sha256']
            and sha256(root/'data-observed/receipt.json')==data['prepared_receipt_sha256']
            and receipt['sha256']==selected_data['data_sha256']==data['prepared_data_sha256'],'Prepared data identities differ')
    require(selected_data['selected_records']==len(selected_data['selected'])==330 and not selected_data['excluded'],
            'Require 330 unique complete examples')
    require(observed['common_identical_records']==220 and observed['observed_replacements']==data['replaced_records']
            and data['replaced_records']+observed['fallback_records']==110,'Contrast coverage differs')
    require(data['coverage_passed'] is (data['replaced_records']>=40 and data['replaced_tasks']>=20),'Coverage rule changed')
    cohorts={'dev':[t['id'] for t in inventory if t['split']=='dev'],
             'historical':[t['id'] for t in inventory if t.get('evaluation_role')=='historical_final_regression_not_selection'],
             'fresh':[t['id'] for t in inventory if t.get('evaluation_role')=='fresh_final_not_selection']}
    require(tuple(map(len,cohorts.values()))==(15,48,24),'Changed cohort sizes')
    training={};campaigns={};costs={};comparisons={};pairs=[];legacy={};selection=None
    within=controller['finished_epoch']<=controller['deadline_epoch']
    if not data['coverage_passed']:
        require(integrity['unselected_final_test_artifacts_absent'] is True and integrity['no_training_or_evaluation_artifacts'] is True,
                'Coverage rejection must precede training and evaluation')
        release={'accepted':False,'failed_gates':['observed_state_coverage'],'final_test_opened':False,'weights_public':False}
    else:
        freeze=read('derived-data-freeze.json');data_pub=read('derived-data-publication.json')
        require(freeze['before_any_optimizer_updates'] and freeze['derived_data_sha256']==sha256(root/'derived-data.json')
                and data_pub['anonymous_files_verified'] and data_pub['ci_conclusion']=='success'
                and data_pub['derived_data_sha256']==freeze['derived_data_sha256'] and data_pub['commit']==freeze['publication_commit'],
                'Exact derived data must be publicly fixed before updates')
        result=read('training-observed/result.json');identity=read('training-observed/identity.json')
        chosen=read('training-observed/selection.json');metrics=read('training-observed/metrics.json')
        require(result['status']=='completed' and result['steps']==len(metrics)==330,'Training incomplete')
        expected=list(selected_data['selected']);random.Random(plan['shared_recipe']['seed']).shuffle(expected)
        schedule=[{'example_sha256':hashlib.sha256(r['id'].encode()).hexdigest(),'tokens':r['tokens'],'supervised_tokens':r['supervised_tokens']} for r in expected]
        require(chosen['schedule']==schedule and [{k:r[k] for k in ['example_sha256','tokens','supervised_tokens']} for r in metrics]==schedule,
                'Actual optimizer examples differ from frozen one-pass order')
        require([r['step'] for r in metrics]==list(range(1,331)) and all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in metrics),
                'Noncontiguous/nonfinite updates')
        require(result['input_tokens']==sum(r['tokens'] for r in metrics)==data['input_tokens']
                and result['supervised_tokens']==sum(r['supervised_tokens'] for r in metrics)==data['supervised_tokens'],'Training token exposure differs')
        require(identity['input_hashes']['data']==data['prepared_data_sha256'] and identity['input_hashes']['data_receipt']==data['prepared_receipt_sha256']
                and identity['input_hashes']['profile']==protocol['frozen_files']['experiments/021-observed-recovery/profile-observed.json'],
                'Training data/profile identity differs')
        require(all(protocol['frozen_files'].get(k)==v for k,v in identity['source_files'].items()),'Executed unfrozen training code')
        require(all(result.get(k) is True for k in ['frozen_sample_unchanged','frozen_gradients_absent','adapter_changed','saved_reload_greedy_ids_equal']),
                'Training update or save/reload check failed')
        tensor=integrity['adapter_tensor_audits']['observed']
        require(tensor['parameters']==result['trainable_parameters'] and tensor['dtypes']==['F32']
                and tensor['file_sha256']==result['adapter_files']['adapter_model.safetensors'],'Actual adapter tensors differ')
        training=result;selection=read('selection.json');audit=read('execution-audit.json')
        require(audit['audit_source_sha256']==sha256(EXP/'audit.py') and audit['selection_recomputed_without_test_scores'] and audit['training']['observed']['checkpoint_verified']
                and audit['training']['observed']['adapter_files']==result['adapter_files'],'Missing checkpoint/selection audit')
        arms=['dev-base','dev-reference020','dev-observed']+(['final-base','final-selected'] if selection['selected'] else [])
        for arm in arms:
            campaign=read('campaigns/'+arm+'.json');campaigns[arm]=campaign
            ids=cohorts['dev'] if arm.startswith('dev-') else [t['id'] for t in inventory if t['split']=='test']
            has_adapter=not arm.endswith('-base')
            costs[arm]=prior.validate_campaign(campaign,ids,protocol,sha256(root/'protocol.json'),has_adapter)
            require(audit['campaign_sha256'][arm]==sha256(root/'campaigns'/(arm+'.json')),'Execution audit belongs to another campaign')
            rows=audit['variants'][arm]
            require([r['id'] for r in rows]==ids and all(r['progress_and_native_turns_verified'] and r['final_source_verified'] for r in rows)
                    and sum(r['input_token_hashes_verified'] for r in rows)==costs[arm]['calls'],'Incomplete request/source audit')
            health=read('artifacts/'+arm+'.json');loading=read('loading/'+arm+'.json')
            hashes=({'adapter_model.safetensors':protocol['reference_adapter']['adapter_model_sha256']} if arm=='dev-reference020' else
                    {k:v for k,v in result['adapter_files'].items() if k.endswith('.safetensors')} if has_adapter else {})
            require(all(health[k]==protocol[k] for k in ['model_id','revision']) and health['adapter_sha256']==hashes
                    and health['adapter'] is has_adapter and health['profile_sha256']==protocol['frozen_files']['experiments/021-observed-recovery/profile-observed.json'],
                    'Wrong served model or adapter')
            require(not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),'Model loading mismatch')
        recomputed=decision.select(campaigns['dev-base'],campaigns['dev-reference020'],campaigns['dev-observed'],cohorts['dev'],plan)
        require({k:v for k,v in selection.items() if k!='final_test_opened'}=={k:v for k,v in recomputed.items() if k!='final_test_opened'},
                'Selection does not follow fixed development rule')
        require(selection['final_test_opened']==(selection['selected']=='observed'),'Unexpected final exposure')
        if selection['selected']:
            for cohort in ['historical','fresh']:
                subset=lambda c:dict(c,results=[r for r in c['results'] if r['id'] in cohorts[cohort]])
                comparisons[cohort]=decision.paired(subset(campaigns['final-base']),subset(campaigns['final-selected']),cohorts[cohort])
            for arm in ['final-base','final-selected']:
                value=read('legacy-'+arm+'.json');check=integrity['legacy_input_audits'][arm]
                require(value['complete'] and len(value['results'])==9 and check['result_sha256']==sha256(root/('legacy-'+arm+'.json'))
                        and all(r['input_verified'] for r in check['results']),'Legacy input/result audit missing')
                legacy[arm]=value['passed']
            export=integrity['export']
            checks={'legacy_passes':legacy['final-selected'],'save_reload_greedy_match':True,
                'rights_and_secret_scan_pass':integrity['rights_and_secret_scan_pass'],
                'all_execution_audits_pass':within and export.get('created') is True and export.get('adapter_files_match_training_and_evaluated_checkpoint') is True,
                'single_training_seed_and_repository_cluster_limits_disclosed':True}
            release=decision.release_gate(comparisons['fresh'],comparisons['historical'],checks,plan)
        else:
            require(integrity['unselected_final_test_artifacts_absent'] is True,'Unselected final test artifacts exist')
            release={'accepted':False,'failed_gates':['development_selection'],'final_test_opened':False,'weights_public':False}
        for label,base,candidate in [('dev-vs-base','dev-base','dev-observed'),('dev-vs-reference020','dev-reference020','dev-observed')]+([('final','final-base','final-selected')] if selection['selected'] else []):
            baseline={r['id']:r for r in campaigns[base]['results']}
            for r in campaigns[candidate]['results']:
                b=baseline[r['id']]
                pairs.append({'comparison':label,'id':r['id'],'repository':r['repository'],
                    'cohort':'dev' if label.startswith('dev') else 'historical' if r['id'] in cohorts['historical'] else 'fresh',
                    'base_passed':b['passed'],'candidate_passed':r['passed'],
                    'outcome':'improved' if r['passed'] and not b['passed'] else 'regressed' if b['passed'] and not r['passed'] else 'both_passed' if r['passed'] else 'both_failed',
                    'base_calls':b['calls'],'candidate_calls':r['calls'],'base_tokens':b['generated_tokens'],'candidate_tokens':r['generated_tokens'],
                    'base_seconds':b['seconds'],'candidate_seconds':r['seconds'],
                    'base_first_edit_call':next((s['call_index'] for s in b['source_progress'] if s['changed_editable_files']),None),
                    'candidate_first_edit_call':next((s['call_index'] for s in r['source_progress'] if s['changed_editable_files']),None),
                    'base_final_modified_files':len(b['modified_source_files']),
                    'candidate_final_modified_files':len(r['modified_source_files']),
                    'base_exit_status':b['exit_status'],'candidate_exit_status':r['exit_status'],
                    'base_grade_reason':b.get('reason'),'candidate_grade_reason':r.get('reason')})
    samples=[]
    for p in sorted(root.glob('telemetry-attempt-*.csv')):
        with p.open() as f:samples.extend(csv.DictReader(f))
    hardware_module=module(EXP/'hardware.py','observed_hardware')
    hardware=hardware_module.summarize(samples)
    failed_controller=read('controller-failed-attempt-1.json')
    hardware['sampler_errors_by_attempt']={str(v['attempt']):v['telemetry']['errors'] for v in [failed_controller,controller]}
    return {'schema_version':1,'data':data,'selection':selection,'training':training,'evaluation_cost':costs,
            'comparisons':comparisons,'pairs':pairs,'legacy':legacy,'release_decision':release,'hardware':hardware,
            'time':{'started_utc':controller['started_utc'],'finished_utc':controller['finished_utc'],
                    'wall_seconds':controller['finished_epoch']-controller['started_epoch'],'within_original_budget':within},
            'limitations':plan['limitations'],'weights_public':False,'report_sha256':sha256(Path(__file__))}


def render(root):
    root=Path(root);value=assess(root);write_json(root/'summary.json',value);write_json(root/'release-decision.json',value['release_decision'])
    if value['pairs']:
        with (root/'paired-outcomes.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(value['pairs'][0]));w.writeheader();w.writerows(value['pairs'])
        regressions=[r for r in value['pairs'] if r['outcome']=='regressed']
        review=['# 021 配对行为观察','',
                '以下只描述已经执行并独立评分的配对结果，不重新运行任务，也不把相关性当作退步的唯一成因。首次编辑来自每次工具执行后的实际源码哈希；曾编辑与最终留下修改分开记录。', '',
                '| 对照 | 任务 | 新候选首次编辑调用 | 最终修改文件数 | 调用数 | 退出状态 | 判分原因 |',
                '| --- | --- | --- | --- | --- | --- | --- |']
        for r in regressions:
            review.append('| '+' | '.join(str(r[k]) for k in ['comparison','id','candidate_first_edit_call',
                'candidate_final_modified_files','candidate_calls','candidate_exit_status','candidate_grade_reason'])+' |')
        if not regressions:review+=['本轮已完成配对没有新增退步；这不等于其他任务不存在退步。']
        review+=['','所有配对（包括改善、不变和退步）见 [paired-outcomes.csv](paired-outcomes.csv)。',
                 '训练损失来自不同训练样本，不能替代独立任务通过率；开发分区已重复使用，模型发布仍取决于事先固定的最终门槛。']
        (root/'behavior-review.md').write_text('\n'.join(review)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt':'one-gpu-lm-021','font.family':'DejaVu Sans'})
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(root/(name+'.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {})
        plt.close(fig)
    samples=[]
    for path in sorted(root.glob('telemetry-attempt-*.csv')):
        with path.open() as f:samples.extend(csv.DictReader(f))
    hardware_module=module(EXP/'hardware.py','observed_hardware_plot')
    fig,axes=plt.subplots(2,1,figsize=(10,5),layout='constrained',sharex=True)
    axes[0].plot(*hardware_module.plot_series(samples,'utilization_percent'),color='#167c80')
    axes[0].set(ylabel='GPU utilization (%)',title='Observed pipeline hardware timeline',ylim=(0,105))
    axes[1].plot(*hardware_module.plot_series(samples,'power_w'),color='#b45309')
    axes[1].set(xlabel='Minutes from first hardware sample',ylabel='Board power (W)')
    save(fig,'hardware-timeline')
    if value['comparisons']:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,name in zip(axes,['historical','fresh']):
            record=value['comparisons'][name];numbers=[record['base_passed'],record['candidate_passed']]
            ax.bar(['base','021 observed'],numbers,color=['#64748b','#167c80'])
            ax.set(title=name+' final cohort',ylabel='Independent grader passes',ylim=(0,record['pairs']+1))
            for i,n in enumerate(numbers):ax.text(i,n+.1,str(n)+'/'+str(record['pairs']),ha='center')
        save(fig,'final-comparison')
    data=value['data'];categories=data['categories'];labels=['wrong_edit','tool_error','read_only','fallback']
    numbers=[categories.get(k,0) for k in labels[:3]]+[110-data['replaced_records']]
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.bar(labels,numbers,color=['#b45309','#c77d28','#167c80','#64748b'])
    ax.set(title='110 replaceable slots; 220 common records unchanged',ylabel='Training examples');save(fig,'observed-coverage')
    if value['training']:
        metrics=json.loads((root/'training-observed/metrics.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,key in zip(axes,['loss','seconds']):
            ax.plot([r['step'] for r in metrics],[r[key] for r in metrics],color='#167c80');ax.set(xlabel='Optimizer update',ylabel=key);ax.grid(alpha=.2)
        save(fig,'training-curves')
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained');arms=['dev-base','dev-reference020','dev-observed']
        ax.bar(['base','020 recovery','021 observed'],[value['evaluation_cost'][a]['passed'] for a in arms],color=['#64748b','#b45309','#167c80'])
        ax.set(title='Reused development cohort; 15 paired tasks',ylabel='Independent grader passes',ylim=(0,16));save(fig,'development-comparison')
    table='\n'.join('| '+name+' | '+str(v['passed'])+'/'+str(v['tasks'])+' | '+str(v['calls'])+' | '+str(v['failed_without_any_edit'])+' | '+str(v['failed_after_edit'])+' |' for name,v in value['evaluation_cost'].items())
    final='\n'.join('| '+name+' | '+str(v['base_passed'])+'/'+str(v['pairs'])+' | '+str(v['candidate_passed'])+'/'+str(v['pairs'])+' | '+str(len(v['gained']))+' | '+str(len(v['regressed']))+' |' for name,v in value['comparisons'].items())
    statement='Passed the fixed experimental adapter gates; public weight upload still requires exact-package verification.' if value['release_decision']['accepted'] else 'Did not pass the fixed release gates. Keep the base; do not publish candidate weights.'
    lines=['# 021 · Observed training-state recovery','',statement,'',
        'This is one supervised repair experiment, not full-parameter pretraining or iterative RL. Rules and all task partitions were public before collection; exact derived inputs were fixed before optimizer updates. Raw tasks, traces, sources and weights are absent from this bundle.','',
        f"Observed replacements: {data['replaced_records']}/110 across {data['replaced_tasks']}/55 training tasks. The first 220 examples are unchanged; unmatched slots retain the exact 020 examples. Full input and target limits are enforced without truncation. Only executed, independently verified corrections are supervised; base actions and teacher-run failing diagnostics are masked history.",'',
        '![Actual replacement coverage](observed-coverage.png)','','![Pipeline hardware samples](hardware-timeline.png)','',
        '| Evaluation arm | Passes | Calls | Failed without edit | Failed after edit |','| --- | --- | --- | --- | --- |',table,'',
        '| Final cohort | Base | Candidate | Improved | Regressed |','| --- | --- | --- | --- | --- |',final or 'Final cohorts were not opened.','','![Final paired success](final-comparison.png)' if value['comparisons'] else '','',
        'The new candidate must gain at least one development task against each of the base and fixed 020 recovery, with at most one regression against each. Final release requires at least three fresh gains, zero fresh or historical regressions, all nine legacy checks and all engineering/source audits. Test scores never choose a candidate.','',
        '![Training diagnostics](training-curves.png)' if value['training'] else 'No optimizer update was started because observed-state coverage failed.',
        '![Development comparison](development-comparison.png)' if value['training'] else '',
        '', 'Training loss is a sequence of different examples, not heldout loss. Equal update/call/token ceilings do not imply equal token exposure or FLOPs. Adapter inference is unmerged, and sequential shared-host timing is not a production throughput comparison.','',
        f"Actual UTC: {value['time']['started_utc']} to {value['time']['finished_utc']}; {value['time']['wall_seconds']/3600:.3f} hours. Original deadline respected: {value['time']['within_original_budget']}. Hardware samples: {value['hardware']['samples']}; maximum gap {value['hardware']['max_gap_seconds']:.2f}s. Observed intervals cover {value['hardware']['interval_coverage_fraction']:.1%} of the first-to-last sample span. Board-energy integration excludes gaps longer than 15s and all attempt boundaries; it is not total pipeline/system energy.",'',
        'Recovery disclosure: the first CPU data builder failed before any update because a checked history list was later mutated. The corrected separate builder only deep-copies that prefix. Original sources, failed evidence, collection and deadline were preserved; no model collection or candidate was rerun. See derivation-recovery.json and the experiment recovery amendment.','',
        'Limitations: one training seed, one greedy collection rollout per training task, constructed defects, reused development, four small Python fresh-repository clusters, historical Python/JavaScript tasks and unknown pretraining overlap. These results do not certify general multilingual coding, long tasks or daily-use readiness. Read-only audits and reporting do not rerun models or change selection gates.','',
        'Public numerical evidence supports recomputing comparisons and plots. Exact end-to-end model replay requires the private task/trajectory inputs; the public repository alone is insufficient. To use your own authorized tasks, create a new admission and experiment freeze using the shared one_gpu_lm modules. Do not substitute inputs inside this frozen run.','',
        '[Complete evidence and decision](summary.json) · [Frozen protocol](protocol.json) · [Source provenance](sources.json)',
        '[Paired behavioral observations](behavior-review.md)' if value['pairs'] else '']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    value=render(a.run);print(json.dumps(value['release_decision']))
