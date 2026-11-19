"""Recreate 035 reports, anonymized rows and figures from audited summaries."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


BEIJING = timezone(timedelta(hours=8))
COLORS = {'base':'#357170', 'control':'#356ca5', 'masked':'#ce6d32', 'reminder':'#9b4e83'}
LABELS = {'base':'原始底座', 'control':'034 对照 LoRA', 'masked':'034 屏蔽 LoRA', 'reminder':'底座 + 提示'}
CATEGORIES = (
    'inspection_shaped_no_source_change', 'exact_regression', 'other_no_source_change',
    'invalid_format', 'explicit_submit', 'no_tool_receipt', 'unexecuted_policy_rejection',
)
CATEGORY_LABELS = (
    'Inspection-shaped', 'Exact regression', 'Other', 'Invalid format',
    'Submit', 'No tool receipt', 'Policy rejection',
)


def load(directory):
    value = json.loads((directory / 'measurements.json').read_text(encoding='utf-8'))
    if not value['passed'] or value['final_test_opened'] or value['optimizer_updates'] != 0:
        raise ValueError('Only audited development runs are reportable')
    return value


def table(path, rows, fields=None):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def stamp(value):
    return datetime.fromtimestamp(value, BEIJING).strftime('%Y-%m-%d %H:%M:%S')


def new_figure(columns=1):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({
        'svg.hashsalt':'one-gpu-lm-035', 'axes.spines.top':False, 'axes.spines.right':False,
        'font.size':10,
    })
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, columns, figsize=(5.25 * columns, 3.9), layout='constrained', squeeze=False)
    return plt, fig, axes[0]


def finish_figure(plt, fig, path):
    fig.savefig(path, metadata={'Date':None})
    plt.close(fig)
    svg = path.read_text(encoding='utf-8')
    path.write_text('\n'.join(line.rstrip() for line in svg.splitlines()) + '\n', encoding='utf-8')


def rate(count, total):
    return f'{count}／{total}'


def ordered_groups(rows, arms, n):
    groups = {arm:sorted((r for r in rows if r['arm']==arm),key=lambda r:r['task_number']) for arm in arms}
    expected = list(range(1,n+1))
    if len(rows) != n*len(arms) or any([r['task_number'] for r in group]!=expected for group in groups.values()):
        raise ValueError('Each registered task must have one audited result per arm')
    return groups


def first_edit(groups, limit, path, ax):
    points = [
        {'call':i, **{arm:sum(r['first_edit_call'] is not None and r['first_edit_call']<=i for r in rows)/len(rows)
                      for arm,rows in groups.items()}}
        for i in range(limit+1)
    ]
    table(path, points)
    for arm in groups:
        ax.step([r['call'] for r in points],[100*r[arm] for r in points],where='post',label=arm,
                color=COLORS[arm],lw=2)
    ax.set(title='Time to first source edit',xlabel='Model call',ylabel='Tasks with a source edit (%)',
           xlim=(0,limit),ylim=(0,103),xticks=range(0,limit+1,2))
    ax.grid(axis='y', alpha=.25)
    ax.legend(frameon=False)


def run_time(value):
    hw = value['hardware']
    elapsed = (value['finished_epoch'] - value['started_epoch'])/60
    return (f"北京时间 {stamp(value['started_epoch'])} 至 {stamp(value['finished_epoch'])}，"
            f"用时 {elapsed:.2f} 分钟。GPU 采样 {hw['samples']:,} 条，采样错误 {hw['sampling_errors']} 次，"
            f"峰值显存 {hw['peak_memory_mib']:,} MiB，最高温度 {hw['max_temperature_c']} °C。")


def outcome_table(lines, arms, totals):
    lines.extend([
        '| 指标 | ' + ' | '.join(LABELS[a] for a in arms) + ' |',
        '|---|' + '---:|'*len(arms),
    ])
    metrics = (
        ('completed_repair','预算内修复且提交'), ('first_exact_regression','首步原样执行指定回归命令'),
        ('no_edit','全程未改源码'), ('correct_without_submission','预算内修复但未提交'),
        ('attempted_calls','模型调用'), ('generated_tokens','生成 token'), ('input_tokens','输入 token'),
        ('timeout_recovered','从回执核算成本的超时题'),
    )
    for key,label in metrics:
        readings = [f'{totals[arm][key]:,}' for arm in arms]
        lines.append('| '+label+' | '+' | '.join(readings)+' |')


def write_rows(path, rows):
    public = []
    for row in rows:
        scalar = {k:v for k,v in row.items() if k != 'before_first_edit_actions'}
        if any(not isinstance(v,(str,int,float,bool,type(None))) for v in scalar.values()):
            raise ValueError('The public table accepts scalar measurements only')
        actions = row['before_first_edit_actions']
        if set(actions)-set(CATEGORIES) or sum(actions.values())!=row['before_first_edit_calls']:
            raise ValueError('Unrecognized or incomplete action categories')
        scalar.update({c:actions.get(c,0) for c in CATEGORIES})
        public.append(scalar)
    fields = list(dict.fromkeys(field for row in public for field in row))
    table(path,public,fields)


def baseline(directory):
    data = load(directory)
    if not data['first_inputs_matched'] or data['matched_tasks']!=20:
        raise ValueError('Expected complete 20-task comparisons')
    arms = ('base','control','masked')
    groups = ordered_groups(data['rows'], arms, 20)
    totals = data['arms']
    write_rows(directory/'matched-rollouts.csv',data['rows'])
    plt, fig, (edit, actions) = new_figure(2)
    first_edit(groups, data['budget']['max_model_calls'], directory/'first-edit-curve.csv', edit)
    offsets = [0] * len(arms)
    for category,label,color in zip(CATEGORIES,CATEGORY_LABELS,
                                    ['#537896','#c87939','#9ba3a7','#ac6585','#7c688c','#b8aa6b','#818862']):
        values = [totals[a]['no_edit_action_counts'].get(category,0) for a in arms]
        if any(values):
            actions.barh(arms,values,left=offsets,label=label,color=color)
        offsets = [previous+new for previous,new in zip(offsets,values)]
    actions.invert_yaxis()
    actions.set(title='Actions in runs with no source edit',xlabel='Calls across those runs')
    if any(offsets):
        actions.legend(frameon=False,fontsize=8)
    finish_figure(plt,fig,directory/'development.svg')
    lines = [
        '# 035 · 原始底座对照和未改码轨迹', '',
        f"同一批 20 道题每个版本跑一次，共 60 条轨迹，三组首次输入 token 逐题一致。本轮新增底座 20 次运行、"
        f"{data['base_verified_generation_requests']} 次生成，另外两组来自 034 已审计的 {data['prior_verified_generation_requests']} 次生成。", '',
    ]
    outcome_table(lines,arms,totals)
    lines += ['', '![First edits and actions in no-edit runs](development.svg)', '',
              '| 未改源码轨迹的特征 | 原始底座 | 034 对照 LoRA | 034 屏蔽 LoRA |','|---|---:|---:|---:|']
    for key,label in (
        ('no_edit_at_call_limit','用完 12 次调用'), ('no_edit_inspection_half_or_more','至少一半调用为查阅形态'),
        ('no_edit_with_repeat','出现相同命令、源码与反馈的重复'), ('no_edit_with_compaction','发生上下文裁剪'),
    ):
        lines.append('| '+label+' | '+' | '.join(f"{totals[a][key]}" for a in arms)+' |')
    lines += ['', '查阅形态仅按已执行命令的保守语法判定。计数区分首次实际源码变更、命令格式错误、非零退出和上下文裁剪，'
              '无法度量每次查阅的质量。这是跨批次开发对照，用于定位下一步干预。', '',
              '底座本轮运行'+run_time(data), '',
              '[逐题数表](matched-rollouts.csv) · [首次改码曲线数据](first-edit-curve.csv) · [审计与硬件汇总](measurements.json)', '']
    (directory/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def reminder(directory):
    data = load(directory)
    analysis = data['analysis']
    if not data['first_inputs_matched'] or analysis['paired_tasks'] != 20:
        raise ValueError('Expected 20 audited reminder pairs')
    arms = ('base','reminder')
    groups = ordered_groups(data['rows'],arms,20)
    total = data['arms']
    write_rows(directory/'matched-rollouts.csv',data['rows'])
    plt,fig,axes = new_figure()
    first_edit(groups,data['budget']['max_model_calls'],directory/'first-edit-curve.csv',axes[0])
    finish_figure(plt,fig,directory/'first-edit.svg')
    lines = [
        '# 035 · 一次性进度提示的同预算对照', '',
        (f"原始底座预算内修复并提交 **{rate(total['base']['completed_repair'],20)}**，"
         f"提示组 **{rate(total['reminder']['completed_repair'],20)}**。"
         f"新增成功 {analysis['improved']} 道，退步 {analysis['regressed']} 道，"
         f"净变化 {analysis['net_completed_repairs']:+d} 道。"), '',
    ]
    outcome_table(lines,arms,total)
    lines += ['', '![Cumulative first edit position](first-edit.svg)', '',
              '| 仓库家族 | 提示组净增完成题数 |','|---|---:|']
    lines += [f'| {family} | {delta:+d} |' for family,delta in analysis['by_family'].items()]
    gate = '通过' if analysis['mechanism_followup_gate_passed'] else '未通过'
    lines += [
        '',
        (f"提示在 {analysis['checkpoint_triggered']} 道题触发。"
         f"在触发前或整条未触发轨迹中，{analysis['unchanged_prompt_different_tasks']} 道题的请求或生成存在差异，"
         f"其中有 {analysis['same_input_different_output_calls']} 次输入 token 相同但输出 token 不同。"
         f"筛到实际触发且触发前输入与输出均匹配的题，提示组新增成功 {analysis['interpretable_triggered_improvements']} 道、"
         f"退步 {analysis['interpretable_triggered_regressions']} 道。"),
        '', f'预注册的机制复验门槛{gate}。这批开发题已用于诊断，结论需要另一批题复验。', '',
        (f"本轮新增 {data['reminder_verified_generation_requests']} 次已核验生成，"
         f"并匹配已审计底座的 {data['base_verified_generation_requests']} 次。"
         '提示只在第 5 次请求可能出现，仍计入该请求的 8,192 token 上限。'),
        '', '提示组运行'+run_time(data), '',
        '[逐题数据](matched-rollouts.csv) · [曲线数据](first-edit-curve.csv) · [审计与硬件汇总](measurements.json)', '',
    ]
    (directory/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['baseline','reminder'])
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    {'baseline':baseline,'reminder':reminder}[args.phase](args.directory)


if __name__=='__main__':
    main()
