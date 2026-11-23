"""Publish aggregate CPU replay measurements from the frozen collection."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path


TARGETS = ('reference_edit', 'final_verification', 'explicit_submission')
FAILURES = {'regression_not_reproduced', 'opening_source_identity', 'opening_changed_source', 'actor_call_budget',
            'actor_token_budget', 'native_action_budget', 'student_sequence_budget', 'actor_time_budget',
            'native_command_roundtrip', 'protected_source_changed', 'unexpected_source_transition',
            'missing_verified_reference_submission', 'protected_grade_failed',
            'audit_actor_budget', 'audit_tool_or_encoding_mismatch', 'audit_submission_or_export_mismatch'}
FAILURES.update('tool_failed_' + kind for kind in (*TARGETS, 'focused_inspection', 'partial_repair_regression'))
FAILURES.update(('reference_hunk_over_budget', 'reference_line_over_budget'))


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def measurements(status, status_sha256, registration, diagnosis=None):
    if not (status['complete'] and status['protocol_sha256'] == registration['protocol_sha256']
            and status['model_generations'] == status['optimizer_updates'] == 0 and not status['final_test_opened']):
        raise ValueError('Require a terminal run of the registered CPU experiment')
    rejected = set()
    if diagnosis:
        if diagnosis['status_sha256'] != status_sha256 or diagnosis['final_test_opened']:
            raise ValueError('Replay diagnostics must match the completed run')
        rejected = {row['case'] for row in diagnosis['case_diagnostics']
                    if row['reason'] == 'Indivisible reference hunk exceeds action token cap'
                    and row['executed_actions'] == 0}
    cases = []
    for row in status['rows']:
        supplied = row.get('failure')
        if row['case'] in rejected and supplied == 'replay_error':
            failure = 'reference_hunk_over_budget'
        else:
            failure = supplied if supplied in FAILURES else 'other_replay_error' if supplied else None
        cases.append({key: row[key] for key in ('case', 'family', 'multifile', 'qualified', 'actor_calls',
                                                'target_counts', 'target_tokens', 'replay_seconds')}
                     | {'failure': failure})
    value = {'protocol_sha256': status['protocol_sha256'], 'status_sha256': status_sha256,
             'start': status['started_epoch'], 'end': status['finished_epoch'], 'cases': cases,
             'total_targets': status['export_rows'], 'gate_passed': status['data_gate_passed'],
             'model_generations': 0, 'optimizer_updates': 0, 'final_test_opened': False}
    if 'retained_episodes' in registration:
        keys = ('retained_episodes', 'rerun_episodes')
        if (any(status[key] != registration[key] for key in keys)
                or status['previous_status_sha256'] != registration['first_status_sha256']):
            raise ValueError('Supplement carryover differs from the published registration')
        value.update({key: status[key] for key in keys})
    return value


def analyze(value, design):
    rows = value['cases']
    if ([row['case'] for row in rows] != list(range(1, design['candidate_episodes'] + 1))
            or len({row['family'] for row in rows}) != design['candidate_families']
            or sum(row['multifile'] for row in rows) != design['candidate_multifile_episodes']
            or not value['start'] < value['end']):
        raise ValueError('Incomplete frozen case population')
    target_counts = Counter()
    qualified = []
    curves = [{'cases_processed': 0, 'qualified_episodes': 0, 'multifile_episodes': 0}]
    for row in rows:
        counts = row['target_counts']
        if not (type(row['qualified']) is bool and type(row['multifile']) is bool and
                0 <= row['actor_calls'] <= design['evaluation_budget']['max_model_calls'] and
                math.isfinite(row['replay_seconds']) and row['replay_seconds'] >= 0):
            raise ValueError('Invalid cost or outcome in replay')
        if row['qualified']:
            if (row['failure'] is not None or set(counts) != set(TARGETS) or
                    counts['reference_edit'] < 1 or counts['final_verification'] != 1 or
                    counts['explicit_submission'] != 1 or
                    not 0 < sum(counts.values()) <= row['actor_calls'] or row['target_tokens'] <= 0):
                raise ValueError('Qualified episode is missing a verified target')
            qualified.append(row)
            target_counts.update(counts)
        elif counts or row['target_tokens'] != 0 or not row['failure']:
            raise ValueError('Incomplete episode cannot contribute training labels')
        curves.append({'cases_processed': row['case'], 'qualified_episodes': len(qualified),
                       'multifile_episodes': curves[-1]['multifile_episodes'] + (row['qualified'] and row['multifile'])})
    multi = [row for row in qualified if row['multifile']]
    gate = design['data_gate']
    coverage = {'episodes': len(qualified), 'families': len({row['family'] for row in qualified}),
                'multifile_episodes': len(multi), 'multifile_families': len({row['family'] for row in multi})}
    checks = {key: count >= gate['minimum_' + key] for key, count in coverage.items()}
    if sum(target_counts.values()) != value['total_targets'] or all(checks.values()) != value['gate_passed']:
        raise ValueError('Collection decision or export accounting differs')
    summary = {'cases': len(rows), 'qualified': coverage, 'targets_by_kind': dict(target_counts),
               'targets': value['total_targets'], 'supervised_tokens': sum(row['target_tokens'] for row in qualified),
               'failures': dict(Counter(row['failure'] for row in rows if row['failure'])),
               'seconds': value['end'] - value['start'], 'gate_checks': checks, 'data_gate_passed': all(checks.values()),
               'model_generations': 0, 'optimizer_updates': 0, 'final_test_opened': False}
    if 'retained_episodes' in value:
        retained, rerun = value['retained_episodes'], value['rerun_episodes']
        if retained + rerun != len(rows) or not 0 <= retained <= len(qualified):
            raise ValueError('Supplement case accounting differs')
        summary.update(retained_episodes=retained, rerun_episodes=rerun, newly_qualified=len(qualified) - retained)
    return summary, curves


def plot(path, curves):
    import matplotlib

    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    matplotlib.rcParams['svg.hashsalt'] = '050-verified-repair-closure'
    fig, axis = plt.subplots(figsize=(7, 3.8), layout='constrained')
    for label, key, color in (('Verified repairs', 'qualified_episodes', '#2D6496'),
                              ('Verified multi-file repairs', 'multifile_episodes', '#B85034')):
        axis.step([row['cases_processed'] for row in curves], [row[key] for row in curves],
                  where='post', linewidth=2, label=label, color=color)
    axis.set(xlabel='Replayed cases', ylabel='Cumulative demonstrations', title='050 CPU replay qualification')
    axis.grid(axis='y', alpha=.25)
    axis.legend(frameon=False)
    fig.savefig(path, metadata={'Date': None})
    plt.close(fig)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')


def render(out, value, design, registration):
    if (value['protocol_sha256'] != registration['protocol_sha256']
            or any(value.get(key) != registration.get(key) for key in ('retained_episodes', 'rerun_episodes'))):
        raise ValueError('Replay protocol does not match the public registration')
    summary, curves = analyze(value, design)
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'measurements.json', value)
    write(out / 'summary.json', summary)
    with (out / 'replay-curve.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=curves[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(curves)
    plot(out / 'replay-curve.svg', curves)
    zone = timezone(timedelta(hours=8))
    start, end = (datetime.fromtimestamp(value[key], zone) for key in ('start', 'end'))
    keep = summary['qualified']
    gate = '通过' if summary['data_gate_passed'] else '未通过'
    supplement = 'rerun_episodes' in summary
    label = '补充重放结果' if supplement else '重放结果'
    scope = (f'保留首轮 {summary["retained_episodes"]} 条示范，重放 {summary["rerun_episodes"]} 条，'
             f'新增 **{summary["newly_qualified"]}** 条。两轮合计 ' if supplement else
             f'共重放 **{summary["cases"]}** 条，')
    lines = [f'# 050 · 历史修复示范{label}', '',
             scope + f'**{keep["episodes"]}** 条通过回归、提交、独立评分和输入重建。'
             f'合格示范覆盖 {keep["families"]} 个家族，其中 {keep["multifile_episodes"]} 条来自 '
             f'{keep["multifile_families"]} 个家族的必要多文件修复。数据门槛**{gate}**。', '',
             f'共保留 **{summary["targets"]}** 个动作和 **{summary["supervised_tokens"]:,}** 个受监督 token。'
             f'北京时间 {start:%H:%M:%S} 至 {end:%H:%M:%S} 在 CPU 上执行，用时 '
             f'{summary["seconds"]:.0f} 秒。', '',
             '| 动作 | 数量 |', '| --- | ---: |',
             *(f'| {name} | {summary["targets_by_kind"].get(kind, 0)} |' for kind, name in
               zip(TARGETS, ('源码编辑', '独立回归', '显式提交'))),
             '', '![累积通过复核的示范数](replay-curve.svg)', '',
             '完整计数、淘汰原因和覆盖门槛见 [summary.json](summary.json)。'
             '匿名逐题测量见 [measurements.json](measurements.json)。', '',
             '这里衡量的是参考修复示范能否重放。模型训练和部署效果分别验收，最终测试集继续封存。']
    over_budget = summary['failures'].get('reference_hunk_over_budget', 0)
    if over_budget:
        lines[-3:-3] = [f'有 {over_budget} 条在动作规划时遇到单个参考补丁片段超出 512 token 上限，'
                       '实际源码编辑调用数为 0。', '']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--diagnosis', type=Path)
    parser.add_argument('--supplement', action='store_true')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    registration_file = 'supplement-registration.json' if args.supplement else 'registration.json'
    design, registration = (read(directory / name) for name in ('design.json', registration_file))
    value = read(args.source)
    if 'status' in value:
        value = measurements(value['status'], value['status_sha256'], registration,
                             read(args.diagnosis) if args.diagnosis else None)
    print(json.dumps(render(args.out, value, design, registration), ensure_ascii=False))
