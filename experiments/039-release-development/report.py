"""Rebuild the 039 decision, paired CSV and model-call curves from the audited rows."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from analysis import ARMS, analyze


BEIJING=timezone(timedelta(hours=8))


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def plot(path, points, labels=('Base weights','031 LoRA')):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'svg.hashsalt':'039-release-development','axes.spines.top':False,
                                'axes.spines.right':False})
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained',sharey=True)
    calls=[p['call'] for p in points]
    for ax,key,title in zip(axes,('edited','completed'),('First source edit','Repair completed and submitted')):
        for arm,color,label in zip(ARMS,('#2D6496','#B85034'),labels):
            ax.step(calls,[100*p[f'{key}_{arm}'] for p in points],where='post',color=color,linewidth=2,label=label)
        ax.set(title=title,xlabel='Model calls',xlim=(0,max(calls)),ylim=(0,101),xticks=calls[::2])
        ax.grid(axis='y',alpha=.25)
    axes[0].set_ylabel('Share of all development runs (%)')
    axes[0].legend(frameon=False,loc='upper left')
    fig.savefig(path,metadata={'Date':None})
    plt.close(fig)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n',encoding='utf-8')


def reproduce_analysis(raw, design):
    result=analyze(raw['rows'],design)
    # CPython 3.12 sums float durations with different rounding than 3.10.
    for arm in ARMS:
        measured=result['costs'][arm]['actor_seconds']
        recorded=raw['analysis']['costs'][arm]['actor_seconds']
        if math.isclose(measured,recorded,rel_tol=0,abs_tol=1e-9):
            result['costs'][arm]['actor_seconds']=recorded
    if result!=raw['analysis'] or not result['complete_accounting']:
        raise ValueError('Exact audited repetitions are required')
    return result


def call_curves(rows, limit):
    groups={arm:[r for r in rows if r['arm']==arm] for arm in ARMS}
    points=[]
    for call in range(limit+1):
        row={'call':call}
        for arm,group in groups.items():
            row[f'edited_{arm}']=sum(r['first_edit_call'] is not None and r['first_edit_call']<=call for r in group)/len(group)
            row[f'completed_{arm}']=sum(r['completed_repair'] and r['attempted_calls']<=call for r in group)/len(group)
        points.append(row)
    return points


def run(directory, design):
    raw=json.loads((directory/'measurements.json').read_text(encoding='utf-8'))
    if not raw['passed'] or raw['final_test_opened'] or raw['weights_public'] or raw['optimizer_updates']:
        raise ValueError('Expected audited development-only measurements')
    result=reproduce_analysis(raw,design)
    rows=raw['rows']
    points=call_curves(rows,design['evaluation_budget']['max_model_calls'])
    write_csv(directory/'call-curves.csv',points)
    plot(directory/'call-curves.svg',points)
    write_csv(directory/'matched-repeats.csv',[
        {key:value for key,value in row.items() if key!='before_first_edit_actions'}
        for row in sorted(rows,key=lambda r:(r['task_number'],r['repetition'],ARMS.index(r['arm'])))
    ])
    base,candidate=(result['successes'][a] for a in ARMS)
    denominator=design['task_count']*design['repetitions']
    lower,upper=result['gain_interval_95_pp']
    status='通过' if result['development_gate_passed'] else '未通过'
    improved=sum(p['gain']>0 for p in result['pairs'])
    regressed=sum(p['gain']<0 for p in result['pairs'])
    lines=['# 039 · 27B 参数独立开发筛查','',
        (f'底座修复并提交 **{base}/{denominator}**，031 LoRA **{candidate}/{denominator}**。'
         f'净变化 **{result["gain_pp"]:+.2f} 个百分点**，家族与任务的配对 95% bootstrap 区间为 '
         f'[{lower:+.2f}, {upper:+.2f}]。开发门槛**{status}**。'),'',
        f'按每道题三轮的完成次数比较，LoRA 在 {improved} 道题上提高，在 {regressed} 道题上下降。','',
        '| 指标 | 底座 | 031 LoRA |','|---|---:|---:|']
    for key,label,group in (
        ('no_edit','全程未编辑','behaviors'),('correct_without_submission','修对未提交','behaviors'),
        ('attempted_calls','调用数','costs'),('input_tokens','输入 token','costs'),('generated_tokens','生成 token','costs'),
        ('timeout_recovered','取回回执的超时','behaviors')):
        lines.append(f"| {label} | {result[group]['base'][key]:,} | {result[group]['candidate'][key]:,} |")
    base_seconds,candidate_seconds=(result['costs'][arm]['actor_seconds'] for arm in ARMS)
    lines.append(f'| 任务执行累计秒数 | {base_seconds:,.2f} | {candidate_seconds:,.2f} |')
    lines+=['','![First edit and completed repair](call-curves.svg)','','| 家族 | 净变化（百分点） |','|---|---:|']
    lines.extend(f'| {family} | {gain:+.2f} |' for family,gain in result['family_gain_pp'].items())
    lines+=['','| 开发门槛 | 结果 |','|---|---:|']
    labels={'complete_audit':'完整审计', 'minimum_gain':'提高至少 10 个百分点',
            'positive_interval_lower_bound':'区间下界大于零','maximum_family_drop':'单家族下降至多 10 个百分点'}
    lines.extend(f"| {labels[key]} | {'满足' if passed else '未满足'} |" for key,passed in result['checks'].items())
    start,end=(datetime.fromtimestamp(raw[k],BEIJING) for k in ('started_epoch','finished_epoch'))
    recovery=raw.get('recovery')
    pause=recovery['resumed_epoch']-recovery['previous_finished_epoch'] if recovery else 0
    restart=recovery.get('restart') if recovery else None
    restart_pause=restart['resumed_epoch']-restart['last_pre_restart_sample_epoch'] if restart else 0
    lines+=['',f'北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}。'
              f'包含 {raw["backend_loads"]} 次加载，按评测窗口和可见采样估算单卡占用 {(end-start).total_seconds()/3600-(pause+restart_pause)/3600:.2f} 小时。'
              f'{raw["verified_requests"]:,} 次生成请求通过回执审计。']
    if recovery:
        lines.extend(['',f'保留首段 {recovery["retained_rollouts"]} 次评测，随后续跑 {recovery["resumed_rollouts"]} 次。'
                     f'曾在首个候选生成前中断，排除等待恢复的 {pause/60:.1f} 分钟。'])
    if restart:
        lines.extend(['',f'工作站重启时已完成 {restart["retained_rollouts"]} 次，恢复后完成其余 {restart["replayed_rollouts"]} 次。'
                     f'从最后一条重启前样本到恢复开始间隔 {restart_pause/60:.1f} 分钟。'
                     f'受中断题额外收到 {restart["interrupted_complete_responses"]} 条完成回执，'
                     f'已知消耗 {restart["known_interrupted_input_tokens"]:,} 输入和 '
                     f'{restart["known_interrupted_generated_tokens"]:,} 生成 token，'
                     f'{restart["interrupted_unknown_responses"]} 条响应成本未知。'
                     '这次中断的生成开销单列，配对只计完整执行。重启前未写入文件的采样错误计数未知。'])
    lines+=['',
        '范围为六个仓库家族的 60 道历史功能修复。每题重复三次，配对区间保留题目内的重复关联。'
        '保护评分器在独立副本重跑原公开测试。预训练去污染、长任务和公开参数验收需要独立证据。','',
        '[匿名逐次结果](matched-repeats.csv) · [曲线数值](call-curves.csv) · [测量与审计](measurements.json)','']
    (directory/'README.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    design=json.loads(Path(__file__).with_name('design.json').read_text(encoding='utf-8'))
    run(p.parse_args().directory,design)
