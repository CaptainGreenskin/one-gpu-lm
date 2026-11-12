"""Recompute 022 comparisons from audited public numbers, without opening raw tasks."""
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
    require(sha256(root/'preparation-plan.json')==protocol['frozen_files']['experiments/022-action-repair/preparation-plan.json'],'Changed fixed rules')
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
    data=read('derived-data.json');observed=read('training-data-audit.json');inputs=read('training-input-audit.json')
    require(observed['audit_source_sha256']==sha256(EXP/'audit_data.py') and inputs['audit_source_sha256']==sha256(EXP/'audit_inputs.py')
            and integrity['script_sha256']==sha256(EXP/'postrun_integrity.py'),'Audit source version differs')
    require(observed['passed'] and observed['derived_data_sha256']==sha256(root/'derived-data.json')
            and observed['prepared_data_sha256']==data['prepared_data_sha256'] and observed['whole_context_and_masking_verified'],
            'Derived data lacks exact input evidence')
    require(inputs['passed'] and inputs['data_sha256']=={'action':data['prepared_data_sha256']}
            and integrity['training_input_audit_sha256']==sha256(root/'training-input-audit.json'),'Actual training-input screen failed')
    require(integrity['training_data_audit_sha256']==sha256(root/'training-data-audit.json'), 'Executed-data audit mismatch')
    selected_data=read('data-action/selection.json');receipt=read('data-action/receipt.json')
    require(sha256(root/'data-action/selection.json')==data['selection_sha256']
            and sha256(root/'data-action/receipt.json')==data['prepared_receipt_sha256']
            and receipt['sha256']==selected_data['data_sha256']==data['prepared_data_sha256'],'Prepared data identities differ')
    require(selected_data['selected_records']==len(selected_data['selected'])==330 and not selected_data['excluded'],
            'Require 330 unique complete examples')
    require(observed['common_identical_records']==275 and observed['early_action_replacements']==data['replaced_records']
            and data['replaced_records']+observed['fallback_records']==55,'Contrast coverage differs')
    require(data['coverage_passed'] is (data['replaced_records']>=40 and all(data['accepted_modes'].get(m,0)>=16 for m in ['evidence_then_edit','missing_read_then_edit'])),'Coverage rule changed')
    cohorts={'dev':[t['id'] for t in inventory if t['split']=='dev'],
             'historical':[t['id'] for t in inventory if t.get('evaluation_role')=='historical_final_regression_not_selection'],
             'fresh':[t['id'] for t in inventory if t.get('evaluation_role')=='fresh_final_not_selection']}
    require(tuple(map(len,cohorts.values()))==(15,48,24),'Changed cohort sizes')
    training={};campaigns={};costs={};comparisons={};pairs=[];legacy={};selection=None
    within=controller['finished_epoch']<=controller['deadline_epoch']
    require(data['coverage_passed'] is True, 'Coverage must pass before model launch')
    require(protocol['exact_training_data_frozen_before_updates'] is True and
            protocol['frozen_files']['runs/reference-data022/derived-data.json']==sha256(root/'derived-data.json') and
            protocol['frozen_files']['training-data-audit.json']==sha256(root/'training-data-audit.json'),
            'Exact derived data must be frozen before model runs')
    result=read('training-action/result.json');identity=read('training-action/identity.json')
    chosen=read('training-action/selection.json');metrics=read('training-action/metrics.json')
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
            and identity['input_hashes']['profile']==protocol['frozen_files']['experiments/022-action-repair/profile-action.json'],
            'Training data/profile identity differs')
    require(all(protocol['frozen_files'].get(k)==v for k,v in identity['source_files'].items()),'Executed unfrozen training code')
    require(all(result.get(k) is True for k in ['frozen_sample_unchanged','frozen_gradients_absent','adapter_changed','saved_reload_greedy_ids_equal']),
            'Training update or save/reload check failed')
    tensor=integrity['adapter_tensor_audits']['action']
    require(tensor['parameters']==result['trainable_parameters'] and tensor['dtypes']==['F32']
            and tensor['file_sha256']==result['adapter_files']['adapter_model.safetensors'],'Actual adapter tensors differ')
    training=result;selection=read('selection.json');audit=read('execution-audit.json')
    require(audit['audit_source_sha256']==sha256(EXP/'audit.py') and audit['selection_recomputed_without_test_scores'] and audit['training']['action']['checkpoint_verified']
            and audit['training']['action']['adapter_files']==result['adapter_files'],'Missing checkpoint/selection audit')
    arms=['dev-base','dev-reference020','dev-action']+(['final-base','final-selected'] if selection['selected'] else [])
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
                and health['adapter'] is has_adapter and health['profile_sha256']==protocol['frozen_files']['experiments/022-action-repair/profile-action.json'],
                'Wrong served model or adapter')
        require(not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),'Model loading mismatch')
    recomputed=decision.select(campaigns['dev-base'],campaigns['dev-reference020'],campaigns['dev-action'],cohorts['dev'],plan)
    require({k:v for k,v in selection.items() if k!='final_test_opened'}=={k:v for k,v in recomputed.items() if k!='final_test_opened'},
            'Selection does not follow fixed development rule')
    require(selection['final_test_opened']==(selection['selected']=='action'),'Unexpected final exposure')
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
    for label,base,candidate in [('dev-vs-base','dev-base','dev-action'),('dev-vs-reference020','dev-reference020','dev-action')]+([('final','final-base','final-selected')] if selection['selected'] else []):
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
    previous=[read(p.name) for p in sorted(root.glob('controller-failed-attempt-*.json'))]
    hardware['sampler_errors_by_attempt']={str(v['attempt']):v['telemetry']['errors'] for v in previous+[controller]}
    cpu=read('cpu-preparation-timing.json')
    require(protocol['frozen_files']['cpu-preparation-timing.json']==sha256(root/'cpu-preparation-timing.json'), 'CPU timing evidence differs')
    # These are descriptive reruns of fixed comparators, never selection inputs.
    prior_hashes={'dev-base': '0ea48a5ef7294ab290dde711220d8a497fd8bd6534eb37770400b59ce688b188', 'dev-reference020': '3bb20b188069b30f36daaeefe0e3ebf550993b40e8f87398010e90fff194afd0'}
    prior_audit_file=EXP.parent/'021-observed-recovery/results/2026-11-12/execution-audit.json'
    require(sha256(prior_audit_file)=='eea9d99b4db06ac331b6e778b8cc475b20bce12ad650cb6ab16c3e86779ddd44', 'Pinned prior execution audit changed')
    prior_audit=json.loads(prior_audit_file.read_text())
    def first_difference(a,b):
        return next((i+1 for i in range(max(len(a),len(b)))
                     if (a[i] if i<len(a) else None)!=(b[i] if i<len(b) else None)),None)
    repeated={}
    for arm,digest in prior_hashes.items():
        previous_file=EXP.parent/'021-observed-recovery/results/2026-11-12/campaigns'/(arm+'.json')
        require(sha256(previous_file)==digest, 'Pinned prior comparator evidence changed')
        old=json.loads(previous_file.read_text())
        paired=decision.paired(old,campaigns[arm],cohorts['dev'])
        repeated[arm]={'prior_campaign_sha256':digest,'current_campaign_sha256':sha256(root/'campaigns'/(arm+'.json')),
            'prior_passed':paired['base_passed'],'current_passed':paired['candidate_passed'],
            'gained':paired['gained'],'regressed':paired['regressed'],
            'selection_input':False,'scope':'Descriptive same-comparator rerun; do not replace current scores or infer a specific cause.'}
        old_rows={r['id']:r for r in prior_audit['variants'][arm]}
        new_rows={r['id']:r for r in audit['variants'][arm]}
        repeated[arm]['changed_task_trace_hashes']=[]
        for ident in paired['gained']+paired['regressed']:
            a,b=old_rows[ident],new_rows[ident]
            repeated[arm]['changed_task_trace_hashes'].append({'id':ident,
                'first_different_input_call':first_difference(a['generation_input_sha256'],b['generation_input_sha256']),
                'first_different_output_call':first_difference(a['generation_output_sha256'],b['generation_output_sha256']),
                'initial_input_equal':a['generation_input_sha256'][0]==b['generation_input_sha256'][0],
                'initial_output_equal':a['generation_output_sha256'][0]==b['generation_output_sha256'][0]})
    feedback=read('rerun-variation-audit.json')
    require(feedback['passed'] and feedback['audit_source_sha256']==sha256(EXP/'audit_reruns.py')
            and feedback['prior_execution_audit_sha256']==sha256(prior_audit_file)
            and feedback['current_execution_audit_sha256']==sha256(root/'execution-audit.json')
            and feedback['model_calls_added']==0 and feedback['inputs_changed'] is False and feedback['causal_effect_tested'] is False,
            'Read-only comparator-feedback inspection identity differs')
    require([r['id'] for r in feedback['observations']]==['fast-deep-equal-map-values','fast-deep-equal-regexp-flags'], 'Changed feedback inspection scope')
    for row in feedback['observations']:
        a=next(r for r in prior_audit['variants']['dev-reference020'] if r['id']==row['id'])
        b=next(r for r in audit['variants']['dev-reference020'] if r['id']==row['id'])
        require(row['old_trajectory_sha256']==a['trajectory_sha256'] and row['current_trajectory_sha256']==b['trajectory_sha256']
                and row['initial_model_input_equal'] is True and row['initial_model_output_equal'] is True
                and row['first_tool_content_equal'] is False and row['first_tool_content_equal_after_timestamp_normalization'] is True,
                'Comparator feedback evidence differs from audited trajectories')
    focus=[]
    for ident in ['fast-deep-equal-map-values','fast-deep-equal-regexp-flags']:
        for arm in ['dev-base','dev-reference020','dev-action']:
            row=next(r for r in campaigns[arm]['results'] if r['id']==ident)
            focus.append({'id':ident,'arm':arm,'passed':row['passed'],'calls':row['calls'],
                          'first_edit_call':next((v['call_index'] for v in row['source_progress'] if v['changed_editable_files']),None),
                          'final_modified_files':len(row['modified_source_files'])})
    return {'schema_version':1,'comparator_feedback_inspection':feedback,'comparator_reruns':repeated,'diagnosed_development_tasks':focus,'cpu_preparation':cpu,'data':data,'selection':selection,'training':training,'evaluation_cost':costs,
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
        review=['# 022 配对行为观察','',
                '以下只描述已经执行并独立评分的配对结果，不重新运行任务，也不把相关性当作退步的唯一成因。首次编辑来自每次工具执行后的实际源码哈希；曾编辑与最终留下修改分开记录。', '',
                '| 开发对照 | 通过 | 未编辑失败 | 编辑后失败 | 无效动作 / 调用 |',
                '| --- | --- | --- | --- | --- |']
        for arm in ['dev-base','dev-reference020','dev-action']:
            cost=value['evaluation_cost'][arm]
            review.append(f"| {arm} | {cost['passed']}/{cost['tasks']} | {cost['failed_without_any_edit']} | {cost['failed_after_edit']} | {cost['format_invalid_calls']}/{cost['calls']} |")
        review+=['',
                '| 对照 | 任务 | 新候选首次编辑调用 | 最终修改文件数 | 调用数 | 退出状态 | 判分原因 |',
                '| --- | --- | --- | --- | --- | --- | --- |']
        for r in regressions:
            review.append('| '+' | '.join('未编辑' if r[k] is None and k=='candidate_first_edit_call' else str(r[k]) for k in ['comparison','id','candidate_first_edit_call',
                'candidate_final_modified_files','candidate_calls','candidate_exit_status','candidate_grade_reason'])+' |')
        review+=['', '021 中两道退步任务在本轮同预算下的结果（已用于诊断的开发题，不是独立最终测试）：', '',
                 '| 任务 | 对照 | 通过 | 首次编辑调用 | 最终修改文件数 | 总调用 |',
                 '| --- | --- | --- | --- | --- | --- |']
        for row in value['diagnosed_development_tasks']:
            review.append('| '+' | '.join('未编辑' if row[k] is None else str(row[k]) for k in
                ['id','arm','passed','first_edit_call','final_modified_files','calls'])+' |')
        if not regressions:review+=['本轮已完成配对没有新增退步；这不等于其他任务不存在退步。']
        audits=json.loads((root/'execution-audit.json').read_text())['variants']['dev-action']
        limited=[(r['id'],r['format_categories'].get('invalid_at_output_limit',0)) for r in audits
                 if r['format_categories'].get('invalid_at_output_limit',0)]
        if limited:
            review+=['', '达到单次生成长度上限且动作解析失败的记录：'+
                     '、'.join(f'{ident}（{count} 次）' for ident,count in limited)+'。这是从已保存输出和解析结果核对出的共现，未增加生成预算验证因果；不能据此推断放宽预算就会修复成功。']
        if regressions and all(r['candidate_first_edit_call'] is None for r in regressions):
            review+=['',f"新增退步共 {len({r['id'] for r in regressions})} 道不同任务，均未发生源码编辑；同一任务可能在两组对照中重复列出。上述格式失败与这些退步任务应分开分析，不能合并成同一个原因。"]
        review+=['','固定对照相较 021 的重新运行（只描述波动，不参与选候选）：','',
                 '| 固定对照 | 021 通过数 | 本轮通过数 | 本轮改善 | 本轮退步 |',
                 '| --- | --- | --- | --- | --- |']
        for arm,paired in value['comparator_reruns'].items():
            review.append(f"| {arm} | {paired['prior_passed']}/15 | {paired['current_passed']}/15 | {', '.join(paired['gained']) or '无'} | {', '.join(paired['regressed']) or '无'} |")
        review+=['','这些跨轮差异不归因给新训练。相同权重与调用上限不保证整条工具轨迹逐字一致；本轮选择只使用本轮完整三组，不能择优使用历史对照。']
        review+=['', '对固定 020 的两道变化题进一步只读检查：首个模型输入和输出均与 021 完全相同；首条工具反馈则不同，但只归一化其中的时间字段后，反馈全文相同。新旧轨迹各含 17 处匹配的日期/时间字段。后续实际模型输入已发生差异；这说明整条工具观察不是逐字固定的。',
                 '这个检查没有改写真实输入、没有补跑模型，也没有隔离时间字段对最终分数的因果影响。不能据此声称时间戳是退步的唯一原因。见 [输入反馈检查回执](rerun-variation-audit.json) 与 [只读检查实现](../../audit_reruns.py)。']
        review+=['','所有配对（包括改善、不变和退步）见 [paired-outcomes.csv](paired-outcomes.csv)。',
                 f"本轮 {value['data']['replaced_records']} 条提前行动替换中，两种训练情境分布为 {value['data']['accepted_modes']}。保留了全部原有失败恢复样本（28 条锚点失败、27 条错误编辑）；这是组合课程，不能单独归因到某一项改动。",
                 '训练损失来自不同训练样本，不能替代独立任务通过率；开发分区已重复使用，模型发布仍取决于事先固定的最终门槛。']
        (root/'behavior-review.md').write_text('\n'.join(review)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt':'one-gpu-lm-022','font.family':'DejaVu Sans'})
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(root/(name+'.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {})
        plt.close(fig)
    samples=[]
    for path in sorted(root.glob('telemetry-attempt-*.csv')):
        with path.open() as f:samples.extend(csv.DictReader(f))
    hardware_module=module(EXP/'hardware.py','observed_hardware_plot')
    fig,axes=plt.subplots(2,1,figsize=(10,5),layout='constrained',sharex=True)
    axes[0].plot(*hardware_module.plot_series(samples,'utilization_percent'),color='#167c80')
    axes[0].set(ylabel='GPU utilization (%)',title='Action-repair pipeline hardware timeline',ylim=(0,105))
    axes[1].plot(*hardware_module.plot_series(samples,'power_w'),color='#b45309')
    axes[1].set(xlabel='Minutes from first hardware sample',ylabel='Board power (W)')
    save(fig,'hardware-timeline')
    if value['comparisons']:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,name in zip(axes,['historical','fresh']):
            record=value['comparisons'][name];numbers=[record['base_passed'],record['candidate_passed']]
            ax.bar(['base','022 action'],numbers,color=['#64748b','#167c80'])
            ax.set(title=name+' final cohort',ylabel='Independent grader passes',ylim=(0,record['pairs']+1))
            for i,n in enumerate(numbers):ax.text(i,n+.1,str(n)+'/'+str(record['pairs']),ha='center')
        save(fig,'final-comparison')
    data=value['data'];categories=data['accepted_modes'];labels=['evidence then edit','missing read then edit','fallback']
    numbers=[categories.get(k,0) for k in ['evidence_then_edit','missing_read_then_edit']]+[55-data['replaced_records']]
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.bar(labels,numbers,color=['#167c80','#b45309','#64748b'])
    ax.set(title='55 replaceable slots; 275 common records unchanged',ylabel='Training examples');save(fig,'action-coverage')
    if value['training']:
        metrics=json.loads((root/'training-action/metrics.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,key in zip(axes,['loss','seconds']):
            ax.plot([r['step'] for r in metrics],[r[key] for r in metrics],color='#167c80');ax.set(xlabel='Optimizer update',ylabel=key);ax.grid(alpha=.2)
        save(fig,'training-curves')
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained');arms=['dev-base','dev-reference020','dev-action']
        ax.bar(['base','020 recovery','022 action'],[value['evaluation_cost'][a]['passed'] for a in arms],color=['#64748b','#b45309','#167c80'])
        for i,arm in enumerate(arms):
            cost=value['evaluation_cost'][arm];ax.text(i,cost['passed']+.2,f"{cost['passed']}/{cost['tasks']}",ha='center')
        ax.set(title='Reused development cohort; 15 paired tasks',ylabel='Independent grader passes',ylim=(0,16));save(fig,'development-comparison')
    arms=['dev-base','dev-reference020','dev-action'];labels=['base','020 recovery','022 action']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for arm,label,color in zip(arms,labels,['#64748b','#b45309','#167c80']):
        campaign=json.loads((root/'campaigns'/(arm+'.json')).read_text())
        first=[next((x['call_index'] for x in r['source_progress'] if x['changed_editable_files']),None) for r in campaign['results']]
        axes[0].step(range(13),[sum(n is not None and n<=k for n in first) for k in range(13)],where='post',label=label,color=color)
    axes[0].set(xlabel='Model calls used',ylabel='Tasks with any source edit',title='Earlier action is not necessarily correct',ylim=(0,16),xticks=[0,2,4,6,8,10,12])
    axes[0].legend();axes[0].grid(alpha=.2)
    for i,arm in enumerate(arms):
        cost=value['evaluation_cost'][arm]
        axes[1].bar(i,cost['failed_without_any_edit'],color='#64748b',label='Failed without edit' if i==0 else None)
        axes[1].bar(i,cost['failed_after_edit'],bottom=cost['failed_without_any_edit'],color='#b45309',label='Failed after edit' if i==0 else None)
    axes[1].set(xticks=range(3),xticklabels=labels,ylabel='Independently graded failures',ylim=(0,16));axes[1].legend()
    save(fig,'action-timing')
    table='\n'.join('| '+name+' | '+str(v['passed'])+'/'+str(v['tasks'])+' | '+str(v['calls'])+' | '+str(v['failed_without_any_edit'])+' | '+str(v['failed_after_edit'])+' |' for name,v in value['evaluation_cost'].items())
    final='\n'.join('| '+name+' | '+str(v['base_passed'])+'/'+str(v['pairs'])+' | '+str(v['candidate_passed'])+'/'+str(v['pairs'])+' | '+str(len(v['gained']))+' | '+str(len(v['regressed']))+' |' for name,v in value['comparisons'].items())
    statement='Passed the fixed experimental adapter gates; public weight upload still requires exact-package verification.' if value['release_decision']['accepted'] else 'Did not pass the fixed release gates. Keep the base; do not publish candidate weights.'
    lines=['# 022 · Earlier evidence-backed repair','',statement,'',
        'This is one supervised repair experiment, not full-parameter pretraining or iterative RL. Rules, inherited task partitions and exact CPU-derived inputs were publicly fixed before any new model rollout or optimizer update. CPU reference preparation is timed separately from the 12-hour model pipeline. Raw tasks, traces, sources and weights are absent from this bundle.','',
        f"Verified early-action replacements: {data['replaced_records']}/55 across {data['replaced_records']}/55 training tasks. The other 275 examples are unchanged; unmatched slots retain the exact 020 examples. Full input and target limits are enforced without truncation. Only executed, independently verified corrections are supervised; reference-run failing diagnostics are masked history.",'',
        '![Actual replacement coverage](action-coverage.png)','','![Pipeline hardware samples](hardware-timeline.png)','',
        '| Evaluation arm | Passes | Calls | Failed without edit | Failed after edit |','| --- | --- | --- | --- | --- |',table,'',
        *((['| Final cohort | Base | Candidate | Improved | Regressed |','| --- | --- | --- | --- | --- |',final,'','![Final paired success](final-comparison.png)']) if value['comparisons'] else ['Final cohorts were not opened.']),'',
        'The new candidate must gain at least one development task against each of the base and fixed 020 recovery, with at most one regression against each. Final release requires at least three fresh gains, zero fresh or historical regressions, all nine legacy checks and all engineering/source audits. Test scores never choose a candidate.','',
        '![Training diagnostics](training-curves.png)' if value['training'] else 'No training evidence available.',
        '![Development comparison](development-comparison.png)' if value['training'] else '',
        '![Editing timing and repair failures](action-timing.png)',
        '', 'Training loss is a sequence of different examples, not heldout loss. Equal update/call/token ceilings do not imply equal token exposure or FLOPs. Adapter inference is unmerged, and sequential shared-host timing is not a production throughput comparison.','',
        f"Actual UTC: {value['time']['started_utc']} to {value['time']['finished_utc']}; {value['time']['wall_seconds']/3600:.3f} hours. Original deadline respected: {value['time']['within_original_budget']}. Hardware samples: {value['hardware']['samples']}; maximum gap {value['hardware']['max_gap_seconds']:.2f}s. Observed intervals cover {value['hardware']['interval_coverage_fraction']:.1%} of the first-to-last sample span. Board-energy integration excludes gaps longer than 15s and all attempt boundaries; it is not total pipeline/system energy.",'',
        f"CPU reference construction took approximately {value['cpu_preparation']['builder_seconds']/60:.2f} minutes before the model pipeline (kernel start tick and completed-status timestamp; roughly one-second boot-epoch uncertainty). CPU audits and publication wait are separate from that construction time.",'',
        'Terminology correction: the 55 preserved late-repair examples comprise 28 failed-anchor recoveries and 27 incorrect-edit recoveries. The frozen overview called them all wrong-edit examples; no data or rule changed. See [the terminology correction](../../ERRATA.md).', '',
        'Limitations: one training seed, constructed defects, reused development, four small Python fresh-repository clusters, historical Python/JavaScript tasks and unknown pretraining overlap. These results do not certify general multilingual coding, long tasks or daily-use readiness. Read-only audits and reporting do not rerun models or change selection gates.','',
        'Public numerical evidence supports recomputing comparisons and plots. Exact end-to-end model replay requires the private task/trajectory inputs; the public repository alone is insufficient. To use your own authorized tasks, create a new admission and experiment freeze using the shared one_gpu_lm modules. Do not substitute inputs inside this frozen run.','',
        '[Complete evidence and decision](summary.json) · [Frozen protocol](protocol.json) · [Source provenance](sources.json)',
        '[Paired behavioral observations](behavior-review.md)' if value['pairs'] else '']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    protocol=json.loads((root/'protocol.json').read_text())
    card=['# 022 编码修复候选说明','',
        '本候选通过固定实验门槛；参数是否已上传应以单独的发布回执为准。' if value['release_decision']['accepted'] else
        '本候选未通过固定发布门槛，未上传 ModelScope；继续保留底座。','',
        '| 项目 | 实际记录 |','| --- | --- |',
        f"| 基模 | {protocol['model_id']} |",f"| 固定版本 | `{protocol['revision']}` |",
        f"| 训练产物 | 未合并 LoRA，{value['training']['trainable_parameters']:,} 个可训练参数 |",
        '| 数据 | 55 道训练任务，4 个仓库，330 条完整示范；275 条原记录、55 条提前编辑并测试的替换 |',
        '| 原失败恢复样本 | 28 条锚点失败 + 27 条错误编辑，全部保留；见[冻结概述的措辞更正](../../ERRATA.md) |',
        f"| 实际监督 | {value['training']['supervised_tokens']:,} token；330 次更新 |",
        '| 推理范围 | 每题 12 次调用，单次 8192 输入 / 512 输出 token，总输出 6144 token |','',
        '| 开发对照 | 独立判分通过 |','| --- | --- |']
    for arm in ['dev-base','dev-reference020','dev-action']:
        cost=value['evaluation_cost'][arm];card.append(f"| {arm} | {cost['passed']}/{cost['tasks']} |")
    card+=['','最终测试已按开发门槛开启，分组结果见 [完整结论](summary.json)。' if value['comparisons'] else
           '开发门槛未通过，本轮最终测试保持封存。','',
           '适用范围：这是单种子、构造缺陷与重复使用开发集上的工具修复 SFT 实验。不能据此认证先进通用编码、多语言长任务或日常生产可用性。未合并适配器的顺序计时也不代表优化后的部署吞吐。','',
           '来源：固定基模按其 Apache-2.0 许可记录；原创实现使用仓库 MIT 许可；训练来源保留各自原始许可与通知。没有使用第三方教师 API。原始任务、工具轨迹、语料与优化器状态保持私有。','',
           '[固定规则](protocol.json) · [逐题配对](paired-outcomes.csv) · [失败与对照波动](behavior-review.md) · [发布决策](release-decision.json)']
    (root/'MODEL_CARD.md').write_text('\n'.join(card)+'\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    value=render(a.run);print(json.dumps(value['release_decision']))
