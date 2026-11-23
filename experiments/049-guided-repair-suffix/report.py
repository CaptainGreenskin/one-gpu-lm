"""Reproduce the public collection decision and plots from anonymous paired measurements."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.verified.decision_state_collection import read, require


DIR = Path(__file__).resolve().parent
ARMS = ('control', 'guided')
BOOL_FIELDS = ('independently_correct', 'verified_and_submitted', 'timeout_recovered')
COUNTS = ('attempted_calls', 'input_tokens', 'generated_tokens', 'invalid_actions')
ADMISSIONS = ('qualified', 'no_verified_edit', 'no_verified_submission', 'unexecuted_target',
              'encoding_budget', 'unapproved_target_content', 'control_only', 'same_outcome')


def summarize(value, registration, design):
    require(value['protocol_sha256'] == registration['protocol_sha256']
            and value['passed'] is value['training_only'] is True
            and value['final_test_opened'] is value['weights_public'] is False
            and value['optimizer_updates'] == 0 and value['started_epoch'] < value['finished_epoch'],
            'Require the complete audited training-only measurement')
    pairs = value['pairs']
    require([pair['pair'] for pair in pairs] == list(range(1, design['training_states'] + 1)),
            'Retain every registered pair')
    outcomes, admissions = Counter(), Counter()
    arms = {arm: Counter() for arm in ARMS}
    for pair in pairs:
        require(set(pair) == {'pair', 'admission', 'qualified_targets', 'arms'}
                and set(pair['arms']) == set(ARMS) and pair['admission'] in ADMISSIONS
                and pair['qualified_targets'] == (3 if pair['admission'] == 'qualified' else 0),
                'Unexpected scope or fields in the anonymous pair')
        for arm, cell in pair['arms'].items():
            require(set(cell) == {*BOOL_FIELDS, *COUNTS, 'first_edit_call', 'actor_seconds'}
                    and all(type(cell[key]) is bool for key in BOOL_FIELDS)
                    and all(type(cell[key]) is int and cell[key] >= 0 for key in COUNTS)
                    and 0 < cell['attempted_calls'] <= design['remaining_calls']
                    and 0 < cell['input_tokens'] and cell['generated_tokens'] <=
                        cell['attempted_calls'] * design['max_generated_tokens_per_call']
                    and cell['invalid_actions'] <= cell['attempted_calls']
                    and (cell['first_edit_call'] is None or
                         type(cell['first_edit_call']) is int and 0 < cell['first_edit_call'] <= cell['attempted_calls'])
                    and (not cell['verified_and_submitted'] or
                         cell['first_edit_call'] is not None and cell['independently_correct'])
                    and math.isfinite(cell['actor_seconds']) and cell['actor_seconds'] >= 0,
                    'Unexpected action counts or cost in the anonymous cell')
            arms[arm].update({key: cell[key] for key in COUNTS})
            arms[arm]['actor_seconds'] += cell['actor_seconds']
            arms[arm]['completed_repairs'] += cell['verified_and_submitted']
            arms[arm]['independently_correct'] += cell['independently_correct']
            arms[arm]['no_edit'] += cell['first_edit_call'] is None
        control, guided = (pair['arms'][arm]['verified_and_submitted'] for arm in ARMS)
        outcome = ('both' if guided else 'control_only') if control else ('guided_only' if guided else 'neither')
        require(pair['admission'] != 'qualified' or outcome == 'guided_only',
                'Require a verified guided-only repair for each admitted episode')
        outcomes[outcome] += 1
        admissions[pair['admission']] += 1
    require(sum(arm['attempted_calls'] for arm in arms.values()) == value['verified_requests']
            and 0 <= value['qualified_families'] <= min(admissions['qualified'], design['training_families']),
            'The complete accounting or qualified coverage differs')
    limits = design['collection_gate']
    net = outcomes['guided_only'] - outcomes['control_only']
    checks = {'qualified_episodes': admissions['qualified'] >= limits['minimum_qualified_episodes'],
              'qualified_families': value['qualified_families'] >= limits['minimum_qualified_families'],
              'net_paired_wins': net >= limits['minimum_net_paired_wins']}
    return {'paired_states': len(pairs), 'physical_rollouts': 2 * len(pairs),
            'verified_requests': value['verified_requests'], 'arms': dict(arms), 'outcomes': dict(outcomes),
            'net_paired_wins': net, 'admissions': dict(admissions), 'qualified_episodes': admissions['qualified'],
            'qualified_targets': 3 * admissions['qualified'], 'qualified_families': value['qualified_families'],
            'collection_gate_checks': checks, 'collection_gate_passed': all(checks.values()),
            'pipeline_seconds': value['finished_epoch'] - value['started_epoch'],
            'optimizer_updates': 0, 'final_test_opened': False, 'weights_public': False}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def points(pairs, limit):
    result = []
    for call in range(limit + 1):
        point = {'new_call': call}
        for arm in ARMS:
            cells = [pair['arms'][arm] for pair in pairs]
            point['first_edit_' + arm] = sum(cell['first_edit_call'] is not None and cell['first_edit_call'] <= call
                                            for cell in cells)
            point['completed_' + arm] = sum(cell['verified_and_submitted'] and cell['attempted_calls'] <= call
                                           for cell in cells)
        result.append(point)
    return result


def plot(path, rows, size):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    matplotlib.rcParams['svg.hashsalt'] = '049-guided-repair-suffix'
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained', sharey=True)
    calls = [row['new_call'] for row in rows]
    for axis, key, title in zip(axes, ('first_edit', 'completed'), ('First source edit', 'Verified repair and submission')):
        for arm, color, label in zip(ARMS, ('#2D6496', '#B85034'), ('Original continuation', 'Guided continuation')):
            axis.step(calls, [row[key + '_' + arm] for row in rows], where='post', color=color, label=label, linewidth=2)
        axis.set(title=title, xlabel='New model calls', xticks=calls, ylim=(0, size + 1))
        axis.grid(axis='y', alpha=.25)
    axes[0].set_ylabel('Training states')
    axes[0].legend(loc='best', frameon=False)
    fig.savefig(path, metadata={'Date': None})
    plt.close(fig)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n', encoding='utf-8')


def render(directory, registration):
    design = read(DIR / 'design.json')
    require(sha256(DIR / 'design.json') == registration['design_sha256'], 'The original frozen design is required')
    value = read(directory / 'measurements.json')
    summary = summarize(value, registration, design)
    write_json(directory / 'summary.json', summary)
    pairs = value['pairs']
    write_csv(directory / 'matched-runs.csv',
              [{'pair': pair['pair'], 'arm': arm, 'admission': pair['admission'], **pair['arms'][arm]}
               for pair in pairs for arm in ARMS])
    rows = points(pairs, design['remaining_calls'])
    write_csv(directory / 'call-curves.csv', rows)
    plot(directory / 'call-curves.svg', rows, len(pairs))
    beijing = timezone(timedelta(hours=8))
    start, end = (datetime.fromtimestamp(value[key], beijing) for key in ('started_epoch', 'finished_epoch'))
    control, guided = (summary['arms'][arm]['completed_repairs'] for arm in ARMS)
    improved = summary['outcomes'].get('guided_only', 0)
    regressed = summary['outcomes'].get('control_only', 0)
    decision = '通过' if summary['collection_gate_passed'] else '未通过'
    lines = ['# 049 · 已查阅源码后的修复接续', '',
             f'在 {len(pairs)} 个配对训练状态上，原续跑修对并提交 **{control}** 次，提示组 **{guided}** 次。'
             f'提示独胜 {improved} 次，原续跑独胜 {regressed} 次。采集扩大门槛**{decision}**。', '',
             f'{summary["qualified_episodes"]} 条完整轨迹符合训练数据准入规则，覆盖 '
             f'{summary["qualified_families"]} 个训练仓库家族，保留 '
             f'{summary["qualified_targets"]} 个编辑、验证或提交动作。', '',
             f'北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}，'
             f'共审计 {summary["physical_rollouts"]} 次执行和 {summary["verified_requests"]} 条生成请求。', '',
             '| 指标 | 原续跑 | 提示组 |', '| --- | ---: | ---: |']
    for label, key in (('全程未改码', 'no_edit'), ('通过独立评分', 'independently_correct'),
                       ('新调用数', 'attempted_calls'), ('新生成 token', 'generated_tokens')):
        left, right = (summary['arms'][arm][key] for arm in ARMS)
        lines.append(f'| {label} | {left:,} | {right:,} |')
    gpu = value['hardware']
    if gpu['samples']:
        lines += ['', f'GPU 采样 {gpu["samples"]} 条，平均利用率 {gpu["utilization_percent"]["mean"]:.1f}%，'
                  f'平均功耗 {gpu["power_w"]["mean"]:.0f} W，峰值显存 {gpu["memory_used_mib"]["maximum"]:.0f} MiB。']
    lines += ['', '![首次改码和验证后提交曲线](call-curves.svg)', '',
              '本轮测量经过筛选的训练状态上的采集效果。模型发布继续按独立开发与最终测试验收。', '',
              '[匿名逐次结果](matched-runs.csv) · [曲线数据](call-curves.csv) · [冻结登记](../../registration.json)']
    (directory / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--registration', type=Path, default=DIR / 'registration.json')
    args = parser.parse_args()
    render(args.results, read(args.registration))
