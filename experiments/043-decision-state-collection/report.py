"""Rebuild the 043 tables and curves from anonymous, audited rollout evidence."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from one_gpu_lm.common.receipts import write_json
from one_gpu_lm.verified.cycle_report import write_csv
from one_gpu_lm.verified.decision_state_collection import read, require


MODES = ('direct', 'low')
KINDS = ('early_edit', 'verified_finish')
CELL_FIELDS = ('audited', 'reason', 'error_type', 'first_edit_call', 'submit_call', 'independently_correct', 'verified_and_submitted',
               'attempted_calls', 'input_tokens', 'generated_tokens', 'actor_seconds', 'grader_seconds',
               'invalid_actions', 'timeout_recovered')
PAIR_FIELDS = ('pair', 'kind', 'family', 'language', 'available_calls', 'mode', *CELL_FIELDS, 'contrast')
CURVE_FIELDS = ('kind', 'metric', 'additional_calls', 'mode', 'count', 'complete_pairs')


def summarize(data):
    pairs = data['pairs']
    require([row['pair'] for row in pairs] == list(range(1, data['expected_pairs'] + 1))
            and data['training_only'] and data['optimizer_updates'] == 0
            and data['final_test_opened'] is False, 'Expected the full recorded training-state cohort')
    summary = {k: data[k] for k in ('training_only', 'optimizer_updates', 'final_test_opened',
                                   'expected_pairs', 'audit_passed', 'journal_requests_verified', 'phase',
                                   'hardware', 'model_id', 'revision', 'registration_commit',
                                   'audit_sha256', 'telemetry_sha256')}
    summary['strata'] = {}
    summary['paired_count'] = 0
    curves, rows = [], []
    for kind in KINDS:
        group = [row for row in pairs if row['kind'] == kind]
        complete = [row for row in group if all(row['modes'][mode]['audited'] for mode in MODES)]
        denominator = len(complete)
        summary['paired_count'] += denominator
        outcomes = {(True, True): 'both', (True, False): 'direct_only',
                    (False, True): 'low_only', (False, False): 'neither'}
        counts = Counter(outcomes[tuple(row['modes'][mode]['verified_and_submitted'] for mode in MODES)]
                         for row in complete)
        payload = {'registered_pairs': len(group), 'complete_pairs': denominator, 'paired_outcomes': dict(counts),
                   'contrast_counts': dict(Counter(row['contrast'] for row in complete)), 'modes': {}}
        if kind == 'verified_finish':
            short = [row for row in complete if row['available_calls'] <= 2]
            payload['one_or_two_calls_left'] = {'pairs': len(short), 'verified_and_submitted':
                {mode: sum(row['modes'][mode]['verified_and_submitted'] for row in short) for mode in MODES}}
        for mode in MODES:
            cells = [case['modes'][mode] for case in complete]
            payload['modes'][mode] = {
                'first_edit': sum(item['first_edit_call'] is not None for item in cells),
                'independently_correct': sum(item['independently_correct'] for item in cells),
                'verified_and_submitted': sum(item['verified_and_submitted'] for item in cells),
                'correct_without_submit': sum(item['independently_correct'] and item['submit_call'] is None
                                              for item in cells),
                **{key: sum(item[key] for item in cells) for key in (
                    'attempted_calls', 'input_tokens', 'generated_tokens', 'actor_seconds',
                    'grader_seconds', 'invalid_actions', 'timeout_recovered')}}
            for case in group:
                rows.append({**{k: case[k] for k in ('pair', 'kind', 'family', 'language',
                                                    'available_calls', 'contrast')},
                              'mode': mode, **{k: case['modes'][mode].get(k) for k in CELL_FIELDS}})
            for call in range(max((row['available_calls'] for row in group), default=0) + 1):
                counts = {'verified_and_submitted': sum(
                    item['verified_and_submitted'] and item['submit_call'] <= call for item in cells)}
                if kind == 'early_edit':
                    counts['first_edit'] = sum(item['first_edit_call'] is not None
                                               and item['first_edit_call'] <= call for item in cells)
                curves.extend(dict(kind=kind, metric=metric, additional_calls=call, mode=mode,
                                   count=count, complete_pairs=denominator) for metric, count in counts.items())
        summary['strata'][kind] = payload
    require(sum(value['registered_pairs'] for value in summary['strata'].values()) == len(pairs)
            and (not data['audit_passed'] or summary['paired_count'] == len(pairs)), 'Unexpected cohort or audit gap')
    return summary, rows, curves


def figure(path, curves):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = '043-decision-state-collection'
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), sharey=True)
    panels = [('early_edit', 'first_edit', 'Early states · first edit'),
              ('early_edit', 'verified_and_submitted', 'Early states · completed repair'),
              ('verified_finish', 'verified_and_submitted', 'Verified states · completed submission')]
    for axis, (kind, metric, title) in zip(axes, panels):
        for mode, color, style in [('direct', '#187876', '-'), ('low', '#a95125', '--')]:
            points = [p for p in curves if (p['kind'], p['metric'], p['mode']) == (kind, metric, mode)]
            if points and points[0]['complete_pairs']:
                axis.step([p['additional_calls'] for p in points],
                          [100 * p['count'] / p['complete_pairs'] for p in points],
                          where='post', color=color, linestyle=style, linewidth=2.1, label=mode)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel('Additional model calls')
        axis.set_ylim(-2, 102)
        axis.grid(axis='y', alpha=.22)
        if axis.lines:
            axis.legend(frameon=False, loc='lower right')
    axes[0].set_ylabel('Share of complete pairs (%)')
    fig.tight_layout()
    fig.savefig(path, metadata={'Date': None})
    plt.close(fig)
    content = path.read_text(encoding='utf-8').splitlines()
    path.write_text('\n'.join(line.rstrip() for line in content) + '\n', encoding='utf-8', newline='\n')


def markdown(summary):
    labels = {'early_edit': '首次改码前', 'verified_finish': '正确补丁已通过回归'}
    lines = ['# 043 · 决策状态配对采集', '',
             f'共核对 **{summary["paired_count"]}/{summary["expected_pairs"]}** 组完整配对，'
             f'逐条校验了 **{summary["journal_requests_verified"]:,}** 条生成回执。'
             '两组从相同源码和可见历史继续，得分要求独立评分通过、验证最终源码并在预算内提交。', '',
             '| 起始状态 | 完整配对 | 关闭思考完成 | 低档思考完成 | 满足规则的动作分叉 |',
             '|---|---:|---:|---:|---:|']
    for kind, label in labels.items():
        row = summary['strata'][kind]
        direct, low = (row['modes'][mode]['verified_and_submitted'] for mode in MODES)
        eligible = row['contrast_counts'].get('matched_executed_contrast', 0)
        lines.append(f'| {label} | {row["complete_pairs"]} | {direct} | {low} | {eligible} |')
    early = summary['strata']['early_edit']['modes']
    finish = summary['strata']['verified_finish']
    short = finish['one_or_two_calls_left']
    lines += ['', f'首次改码前，关闭思考有 {early["direct"]["first_edit"]} 组开始改码，'
              f'低档思考有 {early["low"]["first_edit"]} 组。'
              f'已验证且只剩一到两次调用的 {short["pairs"]} 组中，'
              f'两种模式分别完成 {short["verified_and_submitted"]["direct"]} 和 '
              f'{short["verified_and_submitted"]["low"]} 组提交。', '',
              '![按额外调用次数累计改码和提交](call-curves.svg)', '']
    contrasts = summary['strata']['early_edit']['contrast_counts']
    wrong_edit = contrasts.get('winning_action_not_the_verified_first_edit', 0)
    unmatched = contrasts.get('no_matched_executed_fork', 0)
    if wrong_edit or unmatched:
        lines += [f'有 {wrong_edit} 组的分叉动作未直接写出最终验证正确的补丁，另有 {unmatched} 组未找到'
                  '同时满足可见历史、源码、剩余预算及执行记录的分叉。这些配对保留在完成率统计中。', '']
    lines += [f'首次改码前，关闭思考与低档思考的累计任务用时分别为 '
              f'{early["direct"]["actor_seconds"]/60:.1f} 和 {early["low"]["actor_seconds"]/60:.1f} 分钟。'
              f'在正确补丁已通过回归的状态，两种模式分别用时 '
              f'{finish["modes"]["direct"]["actor_seconds"]/60:.1f} 和 '
              f'{finish["modes"]["low"]["actor_seconds"]/60:.1f} 分钟。', '']
    phase = summary['phase']
    if phase['started_utc'] and phase['finished_utc']:
        def beijing(value):
            return datetime.fromisoformat(value).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')

        lines += [f'北京时间 {beijing(phase["started_utc"])} 开始，{beijing(phase["finished_utc"])} 结束，'
                  f'单卡流程用时 {phase["pipeline_seconds"]/3600:.2f} 小时。', '']
    hardware = summary['hardware']
    if hardware['samples']:
        lines += [f'GPU 记录 {hardware["samples"]:,} 个样本。平均利用率 '
                  f'{hardware["utilization_percent"]["mean"]:.1f}%，平均功耗 {hardware["power_w"]["mean"]:.0f} W，'
                  f'峰值显存 {hardware["memory_used_mib"]["peak"] / 1024:.1f} GiB，'
                  f'最高温度 {hardware["temperature_c"]["peak"]:.0f} °C。'
                  f'记录到 {hardware["collection_errors"]} 次采样错误。', '']
    if not summary['audit_passed']:
        lines += ['全量审计未通过。表格和曲线仅使用两组均通过审计的配对。', '']
    lines += ['本轮衡量已见训练状态上的动作选择，独立开发和最终测试有各自的门槛。', '',
              '[匿名配对结果](pairs.csv) · [曲线数据](call-curves.csv) · [审计摘要](summary.json) · [实验登记](../../design.json)']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    summary, rows, curves = summarize(read(root / 'measurements.json'))
    write_json(root / 'summary.json', summary)
    for name, records, fields in [('pairs.csv', rows, PAIR_FIELDS), ('call-curves.csv', curves, CURVE_FIELDS)]:
        path = root / name
        write_csv(path, records, fields)
        path.write_text(path.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    figure(root / 'call-curves.svg', curves)
    (root / 'README.md').write_text(markdown(summary), encoding='utf-8')


if __name__ == '__main__':
    main()
