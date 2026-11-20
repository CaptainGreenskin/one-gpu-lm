"""Rebuild 038 CSVs, curves and report from the anonymized measurements."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


ARMS = ('prefix', 'compact')
BEIJING = timezone(timedelta(hours=8))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def curves(path, points):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'svg.hashsalt':'one-gpu-lm-038','axes.spines.top':False,'axes.spines.right':False})
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained', sharey=True)
    calls = [p['call'] for p in points]
    for ax, metric, title in zip(axes, ('edited', 'repair'), ('First source edit', 'Repair completed and submitted')):
        for arm, color, label in zip(ARMS, ('#386bab', '#af502d'), ('Original prefix', 'Failure excerpts')):
            ax.step(calls, [100*p[f'{metric}_{arm}'] for p in points], where='post', linewidth=2,
                    color=color, label=label)
        ax.set(title=title, xlabel='Model calls', xlim=(0, max(calls)), ylim=(0, 101), xticks=calls[::2])
        ax.grid(axis='y', alpha=.25)
    axes[0].set_ylabel('Share of all development tasks (%)')
    axes[0].legend(frameon=False, loc='upper left')
    figure.savefig(path, metadata={'Date':None})
    plt.close(figure)
    path.write_text('\n'.join(s.rstrip() for s in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')


def run(directory):
    data = json.loads((directory/'measurements.json').read_text(encoding='utf-8'))
    stats = data['analysis']
    size = stats['paired_tasks']
    if (not data['passed'] or data['final_test_opened'] or data['optimizer_updates']
            or size != 24 or data['opening_change_pairs'] != size-data['preflight']['fallback']):
        raise ValueError('Complete audited 038 measurements required')
    rows = data['rows']
    groups = {arm:sorted((r for r in rows if r['arm']==arm), key=lambda r:r['task_number']) for arm in ARMS}
    if len(rows) != 2*size or any([r['task_number'] for r in g] != list(range(1,size+1)) for g in groups.values()):
        raise ValueError('Unknown or incomplete task pair')
    points = [{'call':call, **{f'edited_{arm}':sum(r['first_edit_call'] is not None
                           and r['first_edit_call']<=call for r in group)/size for arm,group in groups.items()},
               **{f'repair_{arm}':sum(r['completed_repair'] and r['attempted_calls']<=call for r in group)/size
                  for arm,group in groups.items()}}
              for call in range(data['budget']['max_model_calls']+1)]
    write_csv(directory/'call-curves.csv', points)
    curves(directory/'call-curves.svg', points)
    fields = tuple(k for k in rows[0] if k!='before_first_edit_actions')
    categories = sorted({k for row in rows for k in row['before_first_edit_actions']})
    write_csv(directory/'matched-rollouts.csv', [
        {**{k:r[k] for k in fields}, **{k:r['before_first_edit_actions'].get(k,0) for k in categories}}
        for pair in zip(*(groups[arm] for arm in ARMS)) for r in pair])

    base, treated = (data['arms'][a] for a in ARMS)
    advice = '达到预定的独立复验门槛。' if stats['followup_gate_passed'] else '未达到预定的独立复验门槛。'
    lines = ['# 038 · 失败摘录独立复验', '',
             (f"旧格式完成 **{base['completed_repair']}／{size}**，摘录组完成 **{treated['completed_repair']}／{size}**。"
              f"新增 {stats['improvements']} 道，退步 {stats['regressions']} 道，净变化 **{stats['net_repairs']:+d} 道**"
              f"（{100*stats['net_repairs']/size:+.1f} 个百分点）。{advice}"), '',
             '| 指标 | 旧格式 | 失败摘录 |', '|---|---:|---:|']
    for key, label in (('completed_repair','预算内修复并提交'),('no_edit','全程未改源码'),
                       ('correct_without_submission','预算内修复但未提交'),('attempted_calls','模型调用'),
                       ('input_tokens','输入 token'),('generated_tokens','生成 token'),
                       ('timeout_recovered','找回回执的生成超时')):
        lines.append(f'| {label} | {base[key]:,} | {treated[key]:,} |')
    lines += ['', '![First edit and submitted repair curves](call-curves.svg)', '',
              '| 仓库家族 | 配对 | 新增完成 | 退步 | 净增 |', '|---|---:|---:|---:|---:|']
    for family, result in stats['by_family'].items():
        lines.append(f"| {family} | {result['pairs']} | {result['improvements']} | {result['regressions']} | {result['net_repairs']:+d} |")
    lines += ['', '| 预先记录的子组 | 配对 | 旧格式完成 | 摘录组完成 |', '|---|---:|---:|---:|']
    for key, label in (('old_prefix_clipped','旧格式输出被截断'),
                       ('recovered_failure_line','摘录找回旧格式缺失的完整失败行'),('fallback','两臂使用同一旧格式')):
        lines.append(f"| {label} | {stats[key+'_pairs']} | {stats[key+'_repairs_prefix']} | {stats[key+'_repairs_compact']} |")
    proof = data['preflight']
    hardware = data['hardware']
    start, finish = (datetime.fromtimestamp(data[key], BEIJING) for key in ('started_epoch','finished_epoch'))
    shrink = 1-proof['compact_opening_tokens']/proof['prefix_opening_tokens']
    lines += ['',
              (f"从未改码转为改码 {stats['new_edits']} 道，反向变化 {stats['lost_edits']} 道。"
               f"首轮输入由 {proof['prefix_opening_tokens']:,} 降至 {proof['compact_opening_tokens']:,} token，少 {shrink:.1%}。"
               f"整题输入变化 {treated['input_tokens']/base['input_tokens']-1:+.1%}，"
               f"生成 token 变化 {treated['generated_tokens']/base['generated_tokens']-1:+.1%}。"), '',
              (f"北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {finish:%Y-%m-%d %H:%M:%S}，"
               f"耗时 {(finish-start).total_seconds()/60:.2f} 分钟。"
               f"{data['verified_requests']:,} 次生成通过回执审计，GPU 共采样 {hardware['samples']:,} 条，"
               f"错误 {hardware['errors']} 条，峰值显存 {hardware['peak_memory_mib']:,} MiB，"
               f"最高温度 {hardware['max_temperature_c']} °C。"), '',
              '每个仓库家族只做一次双臂运行。这里的独立性基于本项目已知训练与开发来源，底座预训练语料无从核实。'
              '评分用隔离副本重跑同一公开回归测试，结果适用于这四组短任务。摘录同时改变输入内容、顺序与长度。'
              '本轮零参数更新，发布验收未开启。', '',
              '[逐题配对](matched-rollouts.csv) · [曲线数值](call-curves.csv) · [完整审计汇总](measurements.json)', '']
    (directory/'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    run(parser.parse_args().directory)
