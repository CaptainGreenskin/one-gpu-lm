"""Recompute 032 training and three-checkpoint results from public numeric receipts."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from .cycle_report import (pick,ROW_FIELDS,CURVE_FIELDS,UPDATE_FIELDS,HARDWARE_FIELDS,
    PHASE_FIELDS,checked_score,category as legacy_category,phase_cost,hardware,write_csv)
from .historic_eval_design import VARIANTS,analyze,schedule



ROW_FIELDS = (*ROW_FIELDS, 'complete_success_within_budget')

def category(row):
    if checked_score(row) and row['success_within_budget'] and not row.get('submitted'):
        return 'correct_patch_not_submitted'
    return legacy_category(row)

def training_input(result,integrity,curve,updates,telemetry,original):
    safe_integrity=pick(integrity,('schema_version','passed','selected_records','optimizer_updates','last_accumulation_records',
        'record_groups','input_tokens','supervised_tokens','protocol_sha256','receipt_sha256','adapter_files_sha256',
        'actual_tokenizer','audit_model_generations','audit_optimizer_updates','final_test_opened','weights_public'))
    safe_integrity['checks']={k:v for k,v in integrity['checks'].items() if type(v) is bool}
    safe_integrity['mixture']=pick(integrity['mixture'],('full_episodes','corrective_fragments','original_action_records',
        'corrective_action_records','unique_tasks','overlapping_task_ids','selected_records','supervised_tokens',
        'target_truncation','oversampling','thought_labels','new_expert_episodes','new_multifile_episodes','new_expert_action_records','all_031_records_retained_once','new_autonomous_teacher_rollouts'))
    value={'schema_version':1,'candidate':pick(result,(*PHASE_FIELDS,'trainable_parameters')),
      'integrity':safe_integrity,
      'curve':[pick(r,(*CURVE_FIELDS,'record_group_kind')) for r in curve],
      'updates':[pick(r,UPDATE_FIELDS) for r in updates],
      'telemetry':[pick(r,('utc',*HARDWARE_FIELDS)) for r in telemetry],
      'mixed031':pick(original,PHASE_FIELDS),'final_test_opened':False,'weights_public':False}
    value['candidate']['telemetry']=pick(result.get('telemetry',{}),('samples','errors','thread_stopped'))
    return value


def evaluation_input(spec,tasks,audit,controller,telemetry):
    rows=[]
    for r in audit['rows']:
        row=pick(r,(*ROW_FIELDS,'variant','block'))
        row['request_replay']=pick(r.get('request_replay',{}),('action_fingerprint','delivered_feedback_fingerprints'))
        rows.append(row)
    phase=pick(controller,PHASE_FIELDS);phase['telemetry']=pick(controller.get('telemetry',{}),('samples','errors','thread_stopped'))
    phase['backend_lifecycles']=[pick(r,('index','role','variant','started_epoch','ready_epoch','stopped_epoch')) for r in controller.get('backend_lifecycles',[])]
    return {'schema_version':1,'cycle_spec':spec,'tasks':[pick(t,('id','repository','family')) for t in tasks],
      'audit':{'passed':audit.get('passed') is True,'checks':{k:v for k,v in audit.get('checks',{}).items() if type(v) is bool},'rows':rows},
      'evaluation':phase,'telemetry':[pick(r,('utc',*HARDWARE_FIELDS)) for r in telemetry]}


def training_summary(bundle):
    r=bundle['candidate'];old=bundle['mixed031'];curve=bundle['curve'];updates=bundle['updates']
    return {'complete':bool(r.get('complete') and bundle['integrity']['passed']),
      'integrity_checks':bundle['integrity']['checks'],'mixture':bundle['integrity']['mixture'],
      'phase':phase_cost(r),'hardware':hardware(bundle['telemetry'],r),
      'training':pick(r,(*PHASE_FIELDS,'trainable_parameters')),
      'mixed031':pick(old,('training_seconds','optimizer_steps','action_records_consumed','input_tokens','supervised_tokens')),
      'extra_training_seconds':r.get('training_seconds',0)-old.get('training_seconds',0),
      'extra_optimizer_updates':r.get('optimizer_steps',0)-old.get('optimizer_steps',0),
      'record_groups':dict(Counter(x['record_group_kind'] for x in curve)),
      'record_mean_loss_by_group':{g:sum(x['loss'] for x in curve if x['record_group_kind']==g)/sum(x['record_group_kind']==g for x in curve)
          for g in sorted({x['record_group_kind'] for x in curve})},
      'last_accumulation_records':updates[-1]['records'] if updates else None,
      'loss_is_not_validation_quality':True,'final_test_opened':False,'weights_public':False}


def summarize(bundle):
    rows=bundle['audit']['rows'];spec=bundle['cycle_spec'];tasks=bundle['tasks']
    result=analyze(rows,tasks,spec,bundle['audit']['passed'],bundle['audit']['checks'].get('frozen_inputs_unchanged') is True)
    result.update(scope='new_source_family_historical_development_only',conclusion='achieved' if result['development_parameter_gate_passed'] else 'not_achieved',
      phase=phase_cost(bundle['evaluation']),hardware=hardware(bundle['telemetry'],bundle['evaluation']),variants={})
    for v in VARIANTS:
        group=[r for r in rows if r['variant']==v];verified=[r for r in group if checked_score(r)]
        pairs=result['pairs']
        result['variants'][v]={'recorded_runs':len(group),'verified_runs':len(verified),'successes':sum(r['complete_success_within_budget'] for r in verified),'correct_patches_within_budget':sum(r['success_within_budget'] for r in verified),
          'explicit_submissions':sum(bool(r.get('submitted')) for r in verified),'categories':dict(Counter(category(r) for r in group)),
          'invalid_actions':sum(r.get('invalid_actions',0) for r in group),
          'cost':{k:sum(r.get(k,0) for r in group) for k in ['attempted_calls','input_tokens','generated_tokens','actor_seconds','seconds']},
          'repeat_outcome_disagreement_tasks':sum(r['repeat_disagreement'][v] for r in pairs),
          'repeat_action_disagreement_tasks':sum(r['action_disagreement'][v] for r in pairs),
          'unknown_generation_cost_runs':sum(bool(r.get('unknown_inflight_generation_cost')) for r in group)}
    by={r['id']:r for r in result['pairs']};order=[i['task_id'] for i in schedule([t['id'] for t in tasks],3,spec['development']['block_size']) if i['repetition']==2 and i['variant']=='base' and i['task_id'] in by]
    result['accumulation_curve']=[];totals=dict.fromkeys(VARIANTS,0.)
    for n,name in enumerate(order,1):
        for v in VARIANTS:totals[v]+=by[name]['scores'][v]
        result['accumulation_curve'].append({'paired_tasks':n,**{v:totals[v]/n for v in VARIANTS}})
    multi=set(spec['development'].get('multifile_task_ids',[]))
    subset=[r for r in result['pairs'] if r['id'] in multi]
    result['multifile_subset']={'registered_tasks':len(multi),'complete_paired_tasks':len(subset),
        'scores':{v:sum(r['scores'][v] for r in subset)/len(subset) if subset else None for v in VARIANTS}}
    return result


def hardware_figure(path,samples,phase):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,6));hours=[(datetime.fromisoformat(r['utc']).timestamp()-phase['started_epoch'])/3600 for r in samples]
    for ax,key,title in zip(axes.flat,HARDWARE_FIELDS,['Temperature (C)','Power (W)','Utilization (%)','GPU memory (MiB)']):
        ax.plot(hours,[float(r[key]) for r in samples],lw=.7,color='#167d8d');ax.set(xlabel='Hours since pipeline start',ylabel=title)
    fig.suptitle('Recorded GPU samples include loading, CPU tools and idle intervals');fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def report_training(directory,render=True):
    root=Path(directory);b=json.loads((root/'training-input.json').read_text());r=training_summary(b)
    (root/'training-summary.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    write_csv(root/'training-curve.csv',b['curve'],[*CURVE_FIELDS,'record_group_kind']);write_csv(root/'updates.csv',b['updates'],UPDATE_FIELDS)
    if render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(11,6))
        for group,color in [('whole_episode','#167d8d'),('corrective_fragment','#c46b26'),('reference_guided_whole_episode','#6b57a6')]:
            rows=[x for x in b['curve'] if x['record_group_kind']==group]
            axes[0].scatter([x['record'] for x in rows],[x['loss'] for x in rows],s=7 if group=='whole_episode' else 25,color=color,label=group,alpha=.7)
        axes[0].set(xlabel='Record in frozen order',ylabel='Assistant-target loss',title='Different examples; not a validation or ability curve');axes[0].legend()
        axes[1].plot([x['step'] for x in b['updates']],[x['grad_norm'] for x in b['updates']],lw=.8,color='#6b57a6');axes[1].set(xlabel='Optimizer update',ylabel='Gradient norm before clipping')
        fig.tight_layout();fig.savefig(root/'training.png',dpi=160);plt.close(fig)
        hardware_figure(root/'training-hardware.png',b['telemetry'],b['candidate'])
    m=r['mixture'];p=r['phase'];t=r['training']
    lines=['# 032 唯一候选训练核验','',f"训练及逐条审计{'通过' if r['complete'] else '未通过'}。本记录不说明开发能力已经改善。",'',
      f"完整示范 {m['full_episodes']} 条、错误状态片段 {m['corrective_fragments']} 条，共 {m['selected_records']} 条动作、{m['unique_tasks']} 个唯一训练任务。其中新增专家示范 {m.get('new_expert_episodes',0)} 条，多文件示范 {m.get('new_multifile_episodes',0)} 条。仅监督动作，不监督教师思考。",
      f"实际完成 {t['optimizer_steps']} 次更新，最后一组 {r['last_accumulation_records']} 条；监督 token {t['supervised_tokens']:,}，输入 token {t['input_tokens']:,}，可训练参数 {t['trainable_parameters']:,}。",
      f"训练循环 {t['training_seconds']:.2f} 秒，含加载和保存重载探针的流水线 {p['pipeline_seconds']:.2f} 秒（{p['reserved_single_gpu_hours']:.3f} 单卡小时）。相对 031 多 {r['extra_optimizer_updates']} 次更新，训练循环耗时差 {r['extra_training_seconds']:+.2f} 秒；这是两次实际运行的差值，不能单独归因为新增数据内容。",
      f"UTC 起止：{p['started_utc']} → {p['finished_utc']}。",'',
      '核对实际 tokenizer、全部样本及顺序、每组更新数量、有限损失和梯度、冻结参数无梯度、抽样冻结值不变、适配器改变、保存重载训练探针一致。冻结值检查仅抽样三个张量各八个值；两次训练探针各最多16生成 token，只保存输出身份哈希，不能补称精确实际生成量。',
      '损失点来自不同训练样本，图中的下降不能当作验证成功率提高。原始样本、源代码和适配器权重不公开。', '',
      '![训练损失与梯度](training.png)','![训练硬件采样](training-hardware.png)', '',
      '全部检查和硬件采样覆盖见 [training-summary.json](training-summary.json)；完整数字输入见 [training-input.json](training-input.json)。',
      '最终测试保持封存。本轮只允许进入冻结的三组开发对照，不授权发布 ModelScope 参数。', '',
      '复算：`python -m one_gpu_lm.verified.historic_report --training <本目录>`；可加 `--no-figures`。']
    (root/'README.md').write_text('\n'.join(lines)+'\n');return r


def report_evaluation(directory,render=True):
    root=Path(directory);b=json.loads((root/'analysis-input.json').read_text());r=summarize(b)
    (root/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    fields=[*ROW_FIELDS,'variant','block','category'];write_csv(root/'runs.csv',[dict(x,category=category(x)) for x in b['audit']['rows']],fields)
    write_csv(root/'pairs.csv',[{**pick(x,('id','repository','family','gain_vs_base','gain_vs_mixed031')),**x['scores']} for x in r['pairs']],['id','repository','family',*VARIANTS,'gain_vs_base','gain_vs_mixed031'])
    families=r.get('repositories',{});write_csv(root/'families.csv',[{'family':n,'tasks':x['tasks'],**x['scores'],'gain_vs_base':x['gain_vs_base']} for n,x in families.items()],['family','tasks',*VARIANTS,'gain_vs_base'])
    if render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for v,color in zip(VARIANTS,['#777777','#c78b24','#167d8d']):
            c=r['accumulation_curve'];axes[0].plot([x['paired_tasks'] for x in c],[100*x[v] for x in c],label=v,color=color)
        axes[0].set(xlabel='Complete tasks (3 repeats, 3 variants)',ylabel='Success (%)',ylim=(0,105),title='Frozen-order accumulation, not learning');axes[0].legend()
        names=sorted(families);axes[1].barh(names,[100*families[n]['gain_vs_base'] for n in names],color='#167d8d');axes[1].axvline(0,color='#777777',ls='--');axes[1].set(xlabel='Percentage points',title='Multifile032 minus base by repository family')
        fig.tight_layout();fig.savefig(root/'outcomes.png',dpi=160);plt.close(fig)
        hardware_figure(root/'hardware.png',b['telemetry'],b['evaluation'])
    lines=['# 032 三组权重、三轮开发对照','',f"完整配对 {r['paired_tasks']}／{b['cycle_spec']['development']['unique_tasks']} 题；预注册参数收益门槛：{'通过' if r['development_parameter_gate_passed'] else '未通过'}。",'',
      '| 权重 | 正确且提交／运行 | 修对未提交 | 不保留源修改 | 修改未修对 | 结果波动题 | 动作波动题 | 调用 | 生成 token |',
      '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for v,x in r['variants'].items():
        lines.append(f"| {v} | {x['successes']}/{x['verified_runs']} | {x['categories'].get('correct_patch_not_submitted',0)} | {x['categories'].get('no_final_source_edit',0)} | {x['categories'].get('wrong_edit',0)} | {x['repeat_outcome_disagreement_tasks']} | {x['repeat_action_disagreement_tasks']} | {x['cost']['attempted_calls']} | {x['cost']['generated_tokens']} |")
    if 'primary_gain' in r:
        lo,hi=r['primary_interval_95'];lines+=['',f"新候选相对底座 {100*r['primary_gain']:+.2f} 个百分点，家族—任务配对 bootstrap 95% 区间 [{100*lo:+.2f}, {100*hi:+.2f}]；相对 031 候选 {100*r['secondary_gain']:+.2f} 个百分点。三轮重复保留在每题内，不当成更多独立任务。"]
    lines+=['','| 冻结验收项 | 结果 |','| --- | --- |']+[f"| {k} | {'通过' if v else '未通过'} |" for k,v in r.get('checks',{}).items()]
    p=r['phase'];lines+=['',f"开发流水线 {p['reserved_single_gpu_hours']:.3f} 单卡小时，加载 {p['backend_loads']} 次，已完成加载共 {p['completed_backend_load_seconds']:.2f} 秒。所有加载、CPU工具和判分包含在流水线时间内，不能等同GPU计算活跃时间。",
      f"UTC 起止：{p['started_utc']} → {p['finished_utc']}。",'',
      '![完整配对累计结果与家族差异](outcomes.png)','![评测硬件采样](hardware.png)', '',
      '逐次失败与成本见 [runs.csv](runs.csv)，逐题和逐来源配对见 [pairs.csv](pairs.csv)、[families.csv](families.csv)，动作与结果波动、全部输入/输出成本、硬件采样覆盖见 [summary.json](summary.json)。',
      '三者使用同一关闭思考的模型接口、提示、工具、反馈规范化和预算；底座、031与032之间按冻结顺序卸载重载，实际权重哈希按每次轨迹核对。未通过或未完成配对不得支持正向结论。',
      '这些开发任务来自本项目先前未使用的六个仓库，基于真实历史修复。来源家族隔离不等于预训练去污染；公开回归测试可见，保护副本复验不能称为隐藏测试。六个家族、12次调用及一次训练种子不足以证明通用先进编码或长任务能力。公开数字可复算；原始源码、训练样本和任务定义不直接再分发。',
      '032保留031全部数据并增加参考上游补丁的专家示范；定位和目标补丁由参考答案指导，不是教师模型自主发现。三组推理预算相同，训练更新数未匹配，因此不能分离新增数据内容与额外训练计算量的作用。',
      '最终测试继续封存，本轮开发结果不授权上传 ModelScope 参数。', '',
      '复算：`python -m one_gpu_lm.verified.historic_report --results <本目录>`；可加 `--no-figures`。']
    (root/'README.md').write_text('\n'.join(lines)+'\n');return r


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--training',type=Path);g.add_argument('--results',type=Path);p.add_argument('--no-figures',action='store_true');a=p.parse_args()
    v=report_training(a.training,not a.no_figures) if a.training else report_evaluation(a.results,not a.no_figures)
    print(json.dumps(pick(v,('complete','conclusion','paired_tasks','final_test_opened','weights_public'))))
