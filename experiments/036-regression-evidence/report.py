"""Rebuild 036 tables and the first-edit curve from audited public measurements."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


ARMS = ('exit_only', 'transcript')
LABELS = ('仅退出码', '退出码 + 失败输出')
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
        'svg.hashsalt':'one-gpu-lm-036', 'axes.spines.top':False, 'axes.spines.right':False,
        'font.size':10,
    })
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8,4.1),layout='constrained')
    for arm, color, label in zip(ARMS,('#356ca5','#b95937'),('Exit code','Exit code + output')):
        ax.step([r['call'] for r in points], [100*r[arm] for r in points], where='post',
                color=color, linewidth=2, label=label)
    last = points[-1]['call']
    ax.set(title='036 · Time to first source edit', xlabel='Model call',
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
            or samples!=20 or data['opening_token_change_verified_pairs']!=samples):
        raise ValueError('Expected only the complete audited 036 development sweep')
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
        '# 036 · 公开回归输出对照', '',
        (f"仅退出码修复并提交 **{a['completed_repair']}／{samples}**，"
         f"同时提供失败输出 **{b['completed_repair']}／{samples}**。"
         f"新增 {result['new_repairs']} 道，退步 {result['lost_repairs']} 道，"
         f"净变化 **{result['net_repairs']:+d} 道**。"), '',
        '| 指标 | ' + ' | '.join(LABELS) + ' |', '|---|---:|---:|',
    ]
    metrics = (
        ('completed_repair','预算内修复并提交'), ('no_edit','全程未改源码'),
        ('correct_without_submission','预算内修复但未提交'),
        ('first_exact_regression','首步原样执行指定回归命令'),
        ('attempted_calls','模型调用'), ('input_tokens','输入 token'),
        ('generated_tokens','生成 token'), ('timeout_recovered','取回回执的生成超时'),
    )
    for key, label in metrics:
        lines.append(f'| {label} | {a[key]:,} | {b[key]:,} |')
    lines += ['', '![Time to first source edit](first-edit.svg)', '',
              '| 仓库家族 | 追加失败输出的净增完成数 |', '|---|---:|']
    for family,delta in result['net_by_family'].items():
        lines.append(f'| {family} | {delta:+d} |')
    stuck = result['historically_stuck_tasks']
    fields = ('historically_stuck_edited_exit_only','historically_stuck_edited_transcript',
              'historically_stuck_repaired_exit_only','historically_stuck_repaired_transcript')
    edit_a,edit_b,repair_a,repair_b = (result[k] for k in fields)
    gate = '通过' if result['followup_gate_passed'] else '未通过'
    hw = data['hardware']
    length = (data['finished_epoch']-data['started_epoch'])/60
    lines += [
        '',
        (f"035 中两次都未改码的 {stuck} 道题，在本轮分别有 {edit_a}／{stuck} 和 {edit_b}／{stuck} "
         f"修改过源码，最终修复并提交分别为 {repair_a}／{stuck} 和 {repair_b}／{stuck}。"
         f"全组有 {result['new_first_edits']} 道从未改转为改码，"
         f"{result['lost_first_edits']} 道从改码转为未改。"),
        '',
    ]
    gate_rule = data['followup_gate']
    lines.append(
        f"预注册的复验门槛{gate}。规则要求净增至少 {gate_rule['minimum_net_repairs']} 道、"
        f"至少 {gate_rule['minimum_positive_families']} 个家族净增、最多 {gate_rule['maximum_regressions']} 道退步。"
        '本轮沿用已诊断的开发题。'
    )
    lines += [
        '',
        (f"两组共享 {data['public_regressions']} 次预运行，公开测试共耗时 {data['public_tool_seconds']:.2f} 秒。"
         f"其中 {data['clipped_outputs']} 份输出在原有反馈上限处截断，"
         f"{data['received_failure_marker']} 份交付文本含可识别失败标记。"
         '对照同时包含额外上下文长度与具体输出内容的变化。'),
        '',
        (f"从北京时间 {beijing(data['started_epoch'])} 至 {beijing(data['finished_epoch'])}，"
         f"总墙钟用时 {length:.2f} 分钟，"
         f"{data['verified_requests']:,} 次计分生成均有审计回执。"
         f"GPU 采样 {hw['samples']:,} 条，峰值显存 {hw['peak_memory_mib']:,} MiB，"
         f"最高温度 {hw['max_temperature_c']} °C。"),
        '',
    ]
    if 'host_recovery' in data:
        recovery = data['host_recovery']
        split = recovery['retained_rollouts']//2
        prefix = [sum(r['completed_repair'] for r in groups[arm][:split]) for arm in ARMS]
        suffix = [sum(r['completed_repair'] for r in groups[arm][split:]) for arm in ARMS]
        prefix_no_edit = [sum(r['no_edit'] for r in groups[arm][:split]) for arm in ARMS]
        suffix_no_edit = [sum(r['no_edit'] for r in groups[arm][split:]) for arm in ARMS]
        lines += [
            (f"工作站在评估中重启。保留 {recovery['retained_rollouts']} 次已完成运行，"
             f"恢复 {recovery['resumed_rollouts']} 次未计分运行。被中断的一次评估保留了 "
             f"{recovery['completed_responses']} 个完整生成回执及 "
             f"{recovery['known_generated_tokens']:,} 个可计量生成 token。"
             f"另有 {recovery['unknown_responses']} 次已接受请求未生成完成回执，"
             f"未知生成的上限为 {recovery['maximum_unobserved_generation_tokens']} 个 token。"),
            '',
            (f"恢复前记录覆盖 {hw['pre_reboot_observed_minutes']:.2f} 分钟，"
             f"两段之间相隔 {hw['intersegment_gap_minutes']:.2f} 分钟，"
             f"恢复评估历时 {hw['post_reboot_pipeline_minutes']:.2f} 分钟。"
             f"GPU 样本分别为 {hw['pre_reboot_samples']} 和 {hw['post_reboot_samples']} 条，"
             f"恢复段采样错误 {hw['errors']} 次。重启前的采样错误总数未知。"),
            '',
            (f"重启前 {split} 对的修复提交数为 {prefix[0]} 和 {prefix[1]}，恢复后 {samples-split} 对为 "
             f"{suffix[0]} 和 {suffix[1]}。前一段的未改码数为 {prefix_no_edit[0]} 和 {prefix_no_edit[1]}，"
             f"后一段为 {suffix_no_edit[0]} 和 {suffix_no_edit[1]}。"
             '两段题目不同，这些分项只用于检查结论对恢复段的敏感性。'),
            '',
        ]
    else:
        lines.extend([f"GPU 采样错误 {hw['errors']} 次。",''])
    lines += ['[匿名逐题表](matched-rollouts.csv) · [曲线数值](first-edit-curve.csv) · [审计与硬件汇总](measurements.json)', '']
    (directory/'README.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    run(parser.parse_args().directory)
