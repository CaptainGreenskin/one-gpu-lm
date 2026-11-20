"""Rebuild 037 tables and the first-edit curve from audited public measurements."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


ARMS = ('prefix', 'compact')
LABELS = ('旧格式', '失败摘录')
TZ = timezone(timedelta(hours=8))


def beijing(seconds):
    return datetime.fromtimestamp(seconds, TZ).strftime('%Y-%m-%d %H:%M:%S')


def csv_table(path, rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def curve_plot(path, points):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({
        'svg.hashsalt':'one-gpu-lm-037', 'axes.spines.top':False, 'axes.spines.right':False,
        'font.size':10,
    })
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8,4.1),layout='constrained')
    for arm, color, label in zip(ARMS,('#356ca5','#b95937'),('Original prefix','Failure excerpts')):
        ax.step([r['call'] for r in points], [100*r[arm] for r in points], where='post',
                color=color, linewidth=2, label=label)
    last = points[-1]['call']
    ax.set(title='037 · Time to first source edit', xlabel='Model call',
           ylabel='Tasks with a source edit (%)', ylim=(0,103), xlim=(0,last), xticks=range(last+1))
    ax.grid(axis='y',alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(path,metadata={'Date':None})
    plt.close(fig)
    svg = path.read_text(encoding='utf-8')
    path.write_text('\n'.join(line.rstrip() for line in svg.splitlines())+'\n',encoding='utf-8')


def run(directory):
    data = json.loads((directory/'measurements.json').read_text(encoding='utf-8'))
    result = data['analysis']
    samples = result['paired_tasks']
    if (not data['passed'] or data['final_test_opened'] or data['optimizer_updates']!=0
            or samples!=20 or data['opening_change_pairs']!=samples-data['preflight']['fallback']):
        raise ValueError('Expected only the complete audited 037 development sweep')
    limit = data['budget']['max_model_calls']
    groups = {arm:sorted((r for r in data['rows'] if r['arm']==arm),key=lambda r:r['task_number'])
              for arm in ARMS}
    if len(data['rows'])!=len(ARMS)*samples or any(
            [r['task_number'] for r in rows]!=list(range(1,samples+1)) for rows in groups.values()):
        raise ValueError('Expected every arm result for every task')
    points = [
        {'call':i, **{arm:sum(r['first_edit_call'] is not None and r['first_edit_call']<=i
                            for r in rows)/samples for arm,rows in groups.items()}}
        for i in range(limit+1)
    ]
    csv_table(directory/'first-edit-curve.csv',points)
    curve_plot(directory/'first-edit.svg',points)
    fields = tuple(k for k in data['rows'][0] if k!='before_first_edit_actions')
    categories = sorted({k for row in data['rows'] for k in row['before_first_edit_actions']})
    csv_table(directory/'matched-rollouts.csv',[
        {**{k:row[k] for k in fields}, **{k:row['before_first_edit_actions'].get(k,0) for k in categories}}
        for n in range(samples) for row in (groups[arm][n] for arm in ARMS)
    ])
    a,b = (data['arms'][arm] for arm in ARMS)
    lines = [
        '# 037 · 失败摘录对照', '',
        (f"旧格式修复并提交 **{a['completed_repair']}／{samples}**，"
         f"失败摘录组 **{b['completed_repair']}／{samples}**。"
         f"新增 {result['improvements']} 道，退步 {result['regressions']} 道，"
         f"净变化 **{result['net_repairs']:+d} 道**。"), '',
        '| 指标 | ' + ' | '.join(LABELS) + ' |', '|---|---:|---:|',
    ]
    metrics = (
        ('completed_repair','预算内修复并提交'), ('no_edit','全程未改源码'),
        ('correct_without_submission','预算内修复但未提交'),
        ('attempted_calls','模型调用'), ('input_tokens','输入 token'),
        ('generated_tokens','生成 token'), ('timeout_recovered','取回回执的生成超时'),
    )
    for key, label in metrics:
        lines.append(f'| {label} | {a[key]:,} | {b[key]:,} |')
    lines += ['', '![Time to first source edit](first-edit.svg)', '',
              '| 仓库家族 | 失败摘录组净增完成数 |', '|---|---:|']
    for family,delta in result['net_by_family'].items():
        lines.append(f'| {family} | {delta:+d} |')
    lines += ['', '| 固定子组 | 题数 | 旧格式改码 | 摘录组改码 | 旧格式修复提交 | 摘录组修复提交 |',
              '|---|---:|---:|---:|---:|---:|']
    for subset,label in [
        ('old_prefix_clipped','原前缀被截断'),
        ('recovered_failure_line','摘录找回至少一条失败原文'),
        ('historically_stuck','035 中两次均未改码'),
    ]:
        values = [result[subset+'_pairs'], *(
            result[subset+'_'+name+'_'+arm] for name in ('edited','repairs') for arm in ARMS)]
        lines.append('| '+label+' | '+' | '.join(str(value) for value in values)+' |')
    gate = '通过' if result['followup_gate_passed'] else '未通过'
    preflight = data['preflight']
    hw = data['hardware']
    elapsed = (data['finished_epoch']-data['started_epoch'])/60
    shrink = 1-preflight['compact_opening_tokens']/preflight['prefix_opening_tokens']
    lines += [
        '',
        (f"全组 {result['new_edits']} 道从未改转为改码，{result['lost_edits']} 道从改码转为未改。"
         f"预注册复验门槛{gate}。这批题已用于诊断，子组和整体结论均属于开发对照。"),
        '',
        (f"两组均读取同一批 036 已存档输出。旧格式 {preflight['old_prefix_cut']} 份截断。"
         f"摘录保留了 {preflight['recovered_failure_lines_tasks']} 道题的 {preflight['recovered_failure_lines']} 条原格式未包含的完整失败行。"
         f"20 道题开场输入合计由 {preflight['prefix_opening_tokens']:,} 降到 {preflight['compact_opening_tokens']:,} token，少 {shrink:.1%}。"
         '摘录同时改变信息选择、排序及长度。'),
        '',
        (f"整题输入 token 合计变化 {b['input_tokens']/a['input_tokens']-1:+.1%}，"
         f"生成 token 合计变化 {b['generated_tokens']/a['generated_tokens']-1:+.1%}。"
         '这里报告输入和生成的实测量，整体耗时按整次评估记录。'),
        '',
        (f"北京时间 {beijing(data['started_epoch'])} 至 {beijing(data['finished_epoch'])}，"
         f"用时 {elapsed:.2f} 分钟。{data['verified_requests']:,} 次生成均通过回执审计，"
         f"{data['opening_change_pairs']} 对实际开场已和预检 token 哈希对齐。"
         f"GPU 采样 {hw['samples']:,} 条，采样错误 {hw['errors']} 次，峰值显存 {hw['peak_memory_mib']:,} MiB，"
         f"最高温度 {hw['max_temperature_c']} °C。"),
        '',
        '[匿名逐题表](matched-rollouts.csv) · [曲线数值](first-edit-curve.csv) · [审计与硬件汇总](measurements.json)',
        '',
    ]
    (directory/'README.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    run(parser.parse_args().directory)
