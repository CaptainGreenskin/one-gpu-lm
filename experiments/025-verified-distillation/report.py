"""Describe a completed teacher feasibility attempt from sanitized receipts only."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import statistics


def report(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    root = Path(directory)
    controller = json.loads((root / 'controller.json').read_text())
    audit = json.loads((root / 'audit.json').read_text())
    diagnosis = json.loads((root / 'diagnosis.json').read_text())
    calibration = controller.get('collection_policy') == 'explicit_submission_guard_v1'
    rows = controller['tasks']; screened = {r['id']: r for r in audit['rows']}
    samples = list(csv.DictReader((root / 'telemetry.csv').open()))
    duration = controller['finished_epoch'] - controller['started_epoch']
    calls = sum(r['attempted_calls'] for r in rows)
    tokens = sum(r['generated_tokens'] for r in rows)
    passed = sum(r['grade']['passed'] for r in rows)
    eligible = audit['decision']['eligible_full_episodes']
    stamps = [datetime.fromisoformat(r['utc']).timestamp() for r in samples]
    hardware = {'samples': len(samples), 'collection_errors': controller['telemetry']['errors'],
                'maximum_sample_gap_seconds': max((b-a for a,b in zip(stamps,stamps[1:])), default=0),
                'sample_span_seconds': stamps[-1]-stamps[0] if stamps else 0,
                'boundary_uncovered_seconds': (max(0,stamps[0]-controller['started_epoch']) + max(0,controller['finished_epoch']-stamps[-1])) if stamps else duration}
    for key in ['temperature_c', 'power_w', 'utilization_percent', 'memory_used_mib']:
        values = [float(r[key]) for r in samples]
        hardware[key] = {'mean': statistics.mean(values) if values else None, 'peak': max(values) if values else None}
    summary = {'stage': 'training_only_submission_guard_calibration' if calibration else 'training_only_teacher_feasibility', 'attempts': 1, 'controller_complete': controller['complete'],
               'tasks_completed': len(rows), 'tasks_registered': len(audit['rows']), 'independent_grader_passes': passed,
               'eligible_full_episodes': eligible, 'eligible_action_records': sum(len(r.get('encoded_actions',[])) for r in audit['rows'] if r['eligible_for_sft']),
               'attempted_calls_accounted_in_completed_tasks': calls, 'generated_tokens_accounted_in_completed_tasks': tokens,
               'unaccounted_generation_or_not_run_tasks': [r['id'] for r in audit['rows'] if not r['input_replay_passed']],
               'pipeline_seconds': duration, 'task_seconds': sum(r['seconds'] for r in rows),
               'termination_counts': dict(Counter(r['terminal_reason'] for r in rows)), 'hardware': hardware,
               'post_hoc_diagnosis_categories': diagnosis['categories'],
               'explicit_submissions': sum(r['submitted'] for r in rows),
               'unexecuted_policy_rejections': sum(r.get('unexecuted_policy_rejections',0) for r in diagnosis['rows']),
               'decision': audit['decision'], 'optimizer_updates': 0, 'new_development_scores': 0, 'final_test_opened': False}
    (root / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields = ['id','repository','calls','generated_tokens','seconds','first_edit_call','grader_passed','explicit_submit','agent_verified','eligible_for_sft','terminal_reason']
    with (root / 'tasks.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fields); writer.writeheader()
        for r in rows:
            writer.writerow(dict(zip(fields,[r['id'],r['repository'],r['attempted_calls'],r['generated_tokens'],r['seconds'],
                r['first_edit_call'],r['grade']['passed'],r['submitted'],r['agent_verified_final_source'],
                screened[r['id']]['eligible_for_sft'],r['terminal_reason']])))
    labels = [r['id'] for r in rows]; y = list(range(len(rows)))
    colors = ['#167d8d' if screened[r['id']]['eligible_for_sft'] else
              '#d39b32' if r['grade']['passed'] else '#bd573d' for r in rows]
    fig, axes = plt.subplots(1,2,figsize=(12,6),sharey=True)
    axes[0].barh(y,[r['attempted_calls'] for r in rows],color=colors);axes[0].axvline(24,ls='--',color='#555555',lw=1)
    axes[0].set(yticks=y,yticklabels=labels,xlabel='Model calls (cap 24)',title='Each task, one attempt');axes[0].invert_yaxis()
    axes[1].barh(y,[r['generated_tokens'] for r in rows],color=colors)
    axes[1].set(xlabel='Generated tokens',title='Repair outcome and data eligibility')
    axes[1].legend(handles=[Patch(color='#167d8d',label='SFT-eligible episode'),
                           Patch(color='#d39b32',label='Correct repair, not SFT-eligible'),
                           Patch(color='#bd573d',label='Repair not correct')],fontsize=8)
    fig.suptitle('025 teacher feasibility: cost and retained episodes (not model improvement)')
    fig.tight_layout();fig.savefig(root/'task-costs.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));total=0;points=[]
    for r in rows:total+=int(screened[r['id']]['eligible_for_sft']);points.append(total)
    ax.step(range(1,len(rows)+1),points,where='post',color='#167d8d',label='Eligible full episodes')
    ax.axhline(6,color='#bd573d',ls='--',label='Count gate (other gates also required)')
    ax.set(xlabel='Completed training tasks in fixed order',ylabel='Cumulative complete episodes',title='Data yield, not a learning curve',ylim=(-.3,max(7,total+1)))
    ax.legend();fig.tight_layout();fig.savefig(root/'episode-yield.png',dpi=160);plt.close(fig)
    if samples:
        fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
        minutes=[(t-controller['started_epoch'])/60 for t in stamps]
        axes[0].plot(minutes,[float(r['utilization_percent']) for r in samples],lw=.8);axes[0].set(ylabel='GPU utilization (%)',ylim=(0,105))
        axes[1].plot(minutes,[float(r['memory_used_mib'])/1024 for r in samples],lw=.8,color='#167d8d')
        axes[1].set(xlabel='Minutes since controller start',ylabel='GPU memory (GiB)')
        fig.suptitle('Single-attempt hardware samples; includes loading and CPU tool work')
        fig.tight_layout();fig.savefig(root/'hardware.png',dpi=160);plt.close(fig)
    zone=timezone(timedelta(hours=8))
    started=datetime.fromtimestamp(controller['started_epoch'],zone).strftime('%Y-%m-%d %H:%M:%S')
    ended=datetime.fromtimestamp(controller['finished_epoch'],zone).strftime('%Y-%m-%d %H:%M:%S')
    lines=['# 025：提交验证校准结果' if calibration else '# 025：教师轨迹试跑结果','',
        f"本次可行性门槛：**{'通过' if audit['decision']['passed'] else '未通过'}**。独立判分通过 {passed}/{len(audit['rows'])} 题；完整且通过训练编码筛查的轨迹 {eligible} 条。模型参数未训练，不能据此声称学生能力提高。",'',
        f'北京时间 {started} 至 {ended}，控制器持续 {duration/60:.2f} 分钟。共登记 12 道训练题，完成 {len(rows)} 道；本次协议每题尝试一次。'+('此前首次试跑也完整保留，两个协议各自计数。' if calibration else '没有按结果重跑。'),
        f'完整任务累计 {calls} 次调用、{tokens:,} 个生成 token；保留全部失败。存在未完成或在途成本时，见 summary.json 的不完整列表，不能将其成本记为零。','',
        '| 检查 | 结果 |','| --- | --- |']
    lines += [f"| {k} | {'通过' if v else '未通过'} |" for k,v in audit['decision']['checks'].items()]
    lines += ['',f'合格完整轨迹对应 {summary["eligible_action_records"]} 条有效动作编码；动作记录数不当作完整轨迹数。',
        '请求审计使用实际模型 tokenizer 重建每次输入，核对输入/输出 token 摘要和完整目标编码。隐藏判分在新工作区执行，模型不可见参考修复和隐藏用例。','',
        '修复正确与轨迹合格分别计数。原规则要求单独执行完整的指定回归命令；带 cd 前缀或组合命令即使实际运行了检查，也不满足本次严格取证规则。不能据此说模型不会修复，也不回改门槛把旧轨迹算成合格。',
        '后验诊断只描述已执行记录，不改变评分或采样：'+', '.join(f'{k}={v}' for k,v in diagnosis['categories'].items())+'。详见 [diagnosis.json](diagnosis.json)。','',
        '![每题成本与数据资格](task-costs.png)','![完整轨迹累计产出](episode-yield.png)','![硬件采样](hardware.png)','',
        f'硬件共 {len(samples)} 次采样、{hardware["collection_errors"]} 次采集错误，最大相邻采样间隔 {hardware["maximum_sample_gap_seconds"]:.2f} 秒；起止边界未覆盖 {hardware["boundary_uncovered_seconds"]:.2f} 秒。平均利用率含权重加载与 CPU 工具时间，不代表持续训练吞吐。','',
        '局限：仅 3 个训练仓库上的 12 道构造故障，主要是局部修复，不代表真实长任务或独立仓库成功率；公开源码可能已进入底座预训练，不能证明预训练去污染。教师是同一个底座的高预算版本。未比较新开发集，未打开最终测试，未发布权重。','',
        '下一步仍须完成训练数据扩充、仓库家族隔离的新开发集、唯一学生候选及四臂同预算对照。试跑通过只说明采集方法在这批训练题上可行。','',
        f'[固定规则](../../{"calibration-protocol.json" if calibration else "pilot-protocol.json"}) · [逐题数据](tasks.csv) · [完整摘要](summary.json) · [输入和编码审计](audit.json)']
    if calibration:
        comparison = json.loads((root/'comparison.json').read_text())
        pairs = comparison['pairs']
        with (root/'comparison.csv').open('w',newline='') as stream:
            if pairs:
                writer=csv.DictWriter(stream,list(pairs[0]));writer.writeheader();writer.writerows(pairs)
        keys=['correct','submitted','qualified_episode'];names=['Correct repair','Explicit submit','Qualified episode']
        fig,ax=plt.subplots(figsize=(9,4));x=list(range(3))
        for offset,prefix,color,label in [(-.18,'original','#8a9198','Original'),(.18,'guard','#167d8d','Submission guard')]:
            counts=[sum(r[prefix+'_'+key] for r in pairs) for key in keys]
            bars=ax.bar([n+offset for n in x],counts,width=.36,color=color,label=label)
            ax.bar_label(bars)
        ax.set(xticks=x,xticklabels=names,ylabel='Complete paired training tasks',ylim=(0,len(pairs)+2),
               title='Collection feasibility on the same observed tasks')
        ax.legend();fig.tight_layout();fig.savefig(root/'paired-outcomes.png',dpi=160);plt.close(fig)
        summary['paired_analysis']={k:v for k,v in comparison.items() if k not in ['pairs','prefix_audit']}
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        lines += ['', '## 与首次试跑配对', '',
            f'完整且请求审计通过的配对 {comparison["complete_pairs"]} 个；排除 {len(comparison["excluded_pairs"])} 个不完整或未核验配对，普通修复失败仍纳入。两次尝试的所有已观测成本与不确定成本另行保留。',
            f'修复改善 {len(comparison["repair_gains"])} 题，修复退步 {len(comparison["repair_regressions"])} 题。本次明确拒绝、未执行的提交请求 {summary["unexecuted_policy_rejections"]} 次，计入调用和 token 成本，排除出正例监督目标。','',
            '![配对结果](paired-outcomes.png)','',
            '干预反馈前的逐调用输入/生成摘要对照：'+', '.join(f'{k}={v}' for k,v in comparison['prefix_categories'].items())+'。这只能描述本次记录，不能证明未来运行完全确定。',
            '相同训练题按时间先后运行，没有随机化交错，也没有更新模型参数；合格轨迹增加属于采集改进，不能写成独立模型能力提升。','',
            '[逐题配对](comparison.csv) · [配对、全部成本与前缀审计](comparison.json) · [保留的首次试跑](../2026-11-15/)']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();print(json.dumps(report(a.results)['decision']))
