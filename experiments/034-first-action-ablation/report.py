"""Render the public 034 tables and curves from sanitized measurement bundles."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from statistics import mean


ARMS = ('control', 'masked')
COLORS = {'control': '#356ca5', 'masked': '#ce6d32'}
BEIJING = timezone(timedelta(hours=8))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def beijing_stamp(value):
    time = datetime.fromisoformat(value) if isinstance(value, str) else datetime.fromtimestamp(value, timezone.utc)
    return time.astimezone(BEIJING).strftime('%Y-%m-%d %H:%M:%S')


def figure():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({
        'font.size': 10,
        'svg.hashsalt': 'one-gpu-lm-034',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.1), layout='constrained')
    return plt, fig, ax


def save_figure(plt, fig, ax, path):
    ax.grid(axis='y', alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(path, metadata={'Date': None})
    plt.close(fig)
    svg = path.read_text(encoding='utf-8')
    path.write_text('\n'.join(line.rstrip() for line in svg.splitlines()) + '\n', encoding='utf-8')


def training(directory):
    bundle = read(directory / 'measurements.json')
    proof = bundle['integrity']
    if not proof['passed'] or proof['final_test_opened']:
        raise ValueError('Only a complete audited training pair can be reported')
    arms = proof['arms']
    common = arms['control']
    mask = arms['masked']['training']
    curves = {arm: [r for r in bundle['curve'] if r['arm'] == arm] for arm in ARMS}
    for arm, points in curves.items():
        if [r['update'] for r in points] != list(range(1, arms[arm]['updates'] + 1)):
            raise ValueError(f'Missing or repeated optimizer update for {arm}')
    lines = [
        '# 034 · 两组训练完成', '',
        (f"两组按相同顺序处理了 {common['training']['records']:,} 条记录、"
         f"{common['input_tokens']:,} 个输入 token，并各做 {common['updates']:,} 次更新。"
         f"训练审计逐条重算输入和标签，并核对初始 LoRA 指纹。屏蔽组把 {mask['masked_records']} "
         f"条指定首步记录的损失权重设为 0，涉及 {mask['target_supervised_tokens']:,} 个监督 token。"), '',
        '| 组别 | 北京时间开始 | 北京时间结束 | 流水线耗时 | 参与梯度的监督 token | 峰值显存 | 最高温度 | GPU 样本 |',
        '|---|---|---|---:|---:|---:|---:|---:|',
    ]
    for arm, run in arms.items():
        hw = bundle['hardware'][arm]
        lines.append(
            f"| {arm} | {beijing_stamp(run['started_utc'])} | {beijing_stamp(run['finished_utc'])} | "
            f"{run['pipeline_seconds']/60:.2f} 分钟 | {run['training']['effective_supervised_tokens']:,} | "
            f"{hw['peak_memory_mib']:,} MiB | {hw['max_temperature_c']} ℃ | {hw['samples']:,} |"
        )
    if bundle['control_weights_match_032']:
        control = '对照组适配器与 032 历史训练成品的权重字节一致。'
    else:
        control = '对照组适配器的权重哈希与 032 历史成品有差异。本轮组间比较使用同期重训的对照。'
    lines += [
        '', control, '',
        '曲线使用两组均保留梯度的记录。每次更新先取这些记录的原始损失均值，再做 12 次更新的移动平均。'
        '数表另列全记录的原始损失、实际优化目标和梯度范数。开发任务的成功率由单独的成对评估测量。', '',
        '![Training loss on shared supervised targets](training-loss.svg)', '',
        '[逐更新数据](training-curve.csv) · [审计与硬件汇总](measurements.json)', '',
    ]
    write_table(directory / 'training-curve.csv', bundle['curve'])
    plt, fig, ax = figure()
    for arm, points in curves.items():
        losses = [r['common_raw_record_loss'] for r in points]
        smoothed = [mean(losses[max(0, i-11):i+1]) for i in range(len(losses))]
        ax.plot([r['update'] for r in points], smoothed, color=COLORS[arm], linewidth=2, label=arm)
    ax.set(xlabel='Optimizer update', ylabel='Raw record loss (12-update mean)',
           title='034 · Training loss on shared targets')
    save_figure(plt, fig, ax, directory / 'training-loss.svg')
    (directory / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def development(directory):
    bundle = read(directory / 'measurements.json')
    analysis = bundle['analysis']
    rows = bundle['rows']
    pairs = analysis['paired_tasks']
    if (not bundle['passed'] or bundle['final_test_opened'] or not analysis['complete']
            or pairs != 20 or len(rows) != 2 * pairs):
        raise ValueError('Development report requires 20 fully audited pairs')
    totals = analysis['arms']
    delta = 100 * analysis['paired_success_delta']
    lo, hi = (100 * value for value in analysis['descriptive_cluster_bootstrap_95'])
    first = {arm: [r['first_edit_call'] for r in rows if r['arm'] == arm] for arm in ARMS}
    if any(len(values) != pairs or any(v is not None and not 1 <= v <= 12 for v in values)
           for values in first.values()):
        raise ValueError('Invalid first-edit position or missing arm records')
    curve = [
        {'call': k, **{arm: sum(v is not None and v <= k for v in first[arm])/pairs for arm in ARMS}}
        for k in range(13)
    ]
    write_table(directory / 'paired-rollouts.csv', rows)
    write_table(directory / 'first-edit-curve.csv', curve)
    plt, fig, ax = figure()
    for arm in ARMS:
        ax.step([r['call'] for r in curve], [100*r[arm] for r in curve], where='post',
                color=COLORS[arm], linewidth=2, label=arm)
    ax.set(xlabel='Model call', ylabel='Tasks with at least one source edit (%)',
           title='034 · Time to first source edit', ylim=(0, 103), xlim=(0, 12))
    ax.set_xticks(range(13))
    save_figure(plt, fig, ax, directory / 'first-edit.svg')

    lines = [
        '# 034 · 独立开发对照结果', '',
        (f"屏蔽组修复且提交 **{totals['masked']['completed_repair']}／{pairs}**，"
         f"对照组 **{totals['control']['completed_repair']}／{pairs}**，成对差 **{delta:+.1f} 个百分点**。"
         f"屏蔽组独有成功 {analysis['masked_only_success']} 道，对照组独有成功 {analysis['control_only_success']} 道。"), '',
        '| 指标 | 对照组 | 屏蔽组 |', '|---|---:|---:|',
    ]
    metrics = [
        ('completed_repair', '预算内修复且提交'),
        ('first_exact_regression', '首步原样执行指定回归命令'),
        ('no_edit', '从未修改源码'),
        ('correct_without_submission', '预算内修复但未提交'),
        ('attempted_calls', '模型调用数'),
        ('input_tokens', '输入 token 数'),
        ('generated_tokens', '生成 token 数'),
        ('timeout_recovered', '已从回执核算成本的超时任务'),
    ]
    for key, label in metrics:
        lines.append(f"| {label} | {totals['control'][key]:,} | {totals['masked'][key]:,} |")
    lines += ['', '![Cumulative first edit position](first-edit.svg)', '',
              '| 仓库家族 | 屏蔽组净增的完成题数 |', '|---|---:|']
    for family, value in analysis['paired_delta_by_family'].items():
        lines.append(f'| {family} | {value:+d} |')
    hw = bundle['hardware']
    start, finish = bundle['started_epoch'], bundle['finished_epoch']
    counts = {arm: sum(r['multifile'] and r['completed_repair'] for r in rows if r['arm'] == arm) for arm in ARMS}
    multifile = sum(r['multifile'] for r in rows if r['arm'] == 'control')
    gate = '通过' if analysis['mechanism_followup_gate_passed'] else '未通过'
    net = analysis['masked_only_success'] - analysis['control_only_success']
    positive_families = sum(value > 0 for value in analysis['paired_delta_by_family'].values())
    fewer_openings = totals['control']['first_exact_regression'] - totals['masked']['first_exact_regression']
    edits_at_nine = {arm: sum(value is not None and value <= 9 for value in first[arm]) for arm in ARMS}
    lines += [
        '', (f"{multifile} 道多文件题中，对照组完成 {counts['control']} 道，屏蔽组完成 {counts['masked']} 道。"
             f"按仓库家族再按题目分层重抽样，得到描述性 95% 区间 [{lo:+.1f}, {hi:+.1f}] 个百分点。"
             '这轮使用四个家族和一个训练种子。'), '',
        (f"到第 9 次调用时，对照组有 {edits_at_nine['control']} 道题改过源码，屏蔽组有 {edits_at_nine['masked']} 道。"
         '首步原样运行指定命令的计数只比较命令文本。'), '',
        (f'预注册的机制复验门槛{gate}。修复并提交净变化 {net:+d} 道，门槛为至少 +2 道。'
         f'有正收益的仓库家族为 {positive_families} 个，门槛为至少 2 个。原样执行指定首步回归命令减少 {fewer_openings:+d} 次。'), '',
        (f"{len(rows)} 次运行共审计 {bundle['verified_requests']:,} 次生成，核对了每组成对任务的首次输入 token、"
         '原请求、生成回执和保护环境的评分回执。'
         f"评估从北京时间 {beijing_stamp(start)} 运行至 {beijing_stamp(finish)}，用时 {(finish-start)/60:.2f} 分钟。"
         f"模型载入 {bundle['backend_loads']} 次，GPU 采样 {hw['samples']:,} 条，峰值显存 {hw['peak_memory_mib']:,} MiB。"), '',
        '[逐题数表](paired-rollouts.csv)使用匿名题号。[汇总与核验](measurements.json)含规则和结果指纹。最终测试保持封存。', '',
    ]
    (directory / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['training', 'development'])
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    {'training': training, 'development': development}[args.phase](args.directory)
