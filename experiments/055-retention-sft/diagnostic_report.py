"""Render the anonymous, audited 055 retention diagnostic."""

import argparse
from datetime import datetime, timedelta, timezone
from io import StringIO
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory


STRATA = {'prior_regressions': '054 曾退步', 'prior_gains': '054 稳定增益', 'prior_retained': '底座与 054 均通过'}
SERIES = {
    'prior_base': ('Earlier base', '#9ca3af', '--', 'old_base'),
    '054': ('054 LoRA', '#386bab', '-', 'old_054'),
    'rerun_base': ('Rerun base', '#1f2937', '--', 'fresh_base'),
    '055': ('055 LoRA', '#af502d', '-', 'fresh_055'),
}
COUNTS = {'paired_rollouts', 'old_base', 'old_054', 'fresh_base', 'fresh_055', 'fresh_gains', 'fresh_regressions',
          'prior_054_regressions', 'prior_054_gains', 'prior_054_regressions_recovered', 'prior_054_gains_retained'}


def validate(value, registration):
    if (value['protocol_sha256'] != registration['protocol_sha256']
            or registration['task_count'] != 9 or registration['rollout_count'] != 54
            or any(value[name] is not False for name in ('final_test_opened', 'weights_public', 'eligible_for_final_promotion'))
            or not re.fullmatch(r'[0-9a-f]{64}', value['audit_sha256']) or set(value['groups']) != set(STRATA)):
        raise ValueError('Require the complete registered retention diagnostic')
    for row in value['groups'].values():
        if (set(row) != COUNTS or any(type(n) is not int or not 0 <= n <= 9 for n in row.values())
                or row['paired_rollouts'] != 9
                or row['fresh_base'] + row['fresh_gains'] - row['fresh_regressions'] != row['fresh_055']
                or row['old_base'] + row['prior_054_gains'] - row['prior_054_regressions'] != row['old_054']
                or row['prior_054_regressions_recovered'] > row['prior_054_regressions']
                or row['prior_054_gains_retained'] > row['prior_054_gains']):
            raise ValueError('Paired diagnostic counts disagree')
    points = value['submissions_by_call']
    if (len(points) < 2 or [row['calls'] for row in points] != list(range(len(points)))
            or any(set(row) != {'calls', *SERIES} for row in points)):
        raise ValueError('The recorded call thresholds must be ordered and complete')
    for key, (_, _, _, field) in SERIES.items():
        y = [row[key] for row in points]
        if (any(type(n) is not int or not 0 <= n <= 27 for n in y) or y != sorted(y)
                or y[-1] != sum(row[field] for row in value['groups'].values())):
            raise ValueError('A cumulative curve differs from its audited score')
    start, end = (value[key] for key in ('started_epoch', 'finished_epoch'))
    if (not all(math.isfinite(t) for t in (start, end)) or end <= start
            or value['elapsed_seconds'] != end-start or type(value['repository_requests']) is not int
            or value['repository_requests'] < 54):
        raise ValueError('Expected the recorded wall interval and counted requests')


def figure(points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    calls = [row['calls'] for row in points]
    content = StringIO()
    with matplotlib.rc_context({'svg.hashsalt': 'one-gpu-lm-055-diagnostic', 'axes.spines.top': False,
                                'axes.spines.right': False}):
        chart, axis = plt.subplots(figsize=(7.2, 4.3), layout='constrained')
        try:
            for name, (label, color, style, _) in SERIES.items():
                axis.step(calls, [row[name] for row in points], where='post', label=label, color=color,
                          linestyle=style, linewidth=1.8)
            axis.set(xlabel='Model calls used', ylabel='Successful submissions (out of 27)', xlim=(0, calls[-1]),
                     ylim=(0, 28), title='055 retention diagnostic on the 054 development cases')
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(axis='y', alpha=.22)
            axis.legend(frameon=False, loc='upper left')
            chart.savefig(content, format='svg', metadata={'Date': None})
        finally:
            plt.close(chart)
    return '\n'.join(line.rstrip() for line in content.getvalue().splitlines()) + '\n'


def render(value, registration, output):
    validate(value, registration)
    zone = timezone(timedelta(hours=8))
    start, end = (datetime.fromtimestamp(value[key], zone) for key in ('started_epoch', 'finished_epoch'))
    lines = ['# 055 · 底座能力保留诊断', '', '9 道旧开发题各复跑三轮。复跑底座和 055 使用相同的工具、调用与 token 预算。', '',
             '| 任务组 | 配对 | 原底座 | 054 | 复跑底座 | 055 | 复跑新增 | 复跑退步 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, label in STRATA.items():
        row = value['groups'][name]
        counts = [row[key] for key in ('paired_rollouts', 'old_base', 'old_054', 'fresh_base', 'fresh_055',
                                      'fresh_gains', 'fresh_regressions')]
        lines.append('| ' + ' | '.join([label, *map(str, counts)]) + ' |')
    totals = {field: sum(row[field] for row in value['groups'].values()) for field in COUNTS}
    lines.extend(['', f'055 找回原来 054 退步的 {totals["prior_054_regressions_recovered"]}／{totals["prior_054_regressions"]} 个配对，'
                  f'保留原来 054 新增成功的 {totals["prior_054_gains_retained"]}／{totals["prior_054_gains"]} 个配对。', '',
                  '![按实际调用次数累计成功提交](submissions-by-call.svg)', '',
                  f'北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}。'
                  f'复跑单卡流水线用时 {value["elapsed_seconds"]/3600:.2f} 小时，审计核验 {value["repository_requests"]:,} 条请求。', '',
                  '以上用于追踪已知退步与增益的复现。晋级由独立的六家族开发对照决定。', '',
                  '[匿名汇总](summary.json) · [诊断登记](../../diagnostic-registration.json)', ''])
    summary = {key: value[key] for key in ('groups', 'protocol_sha256', 'audit_sha256', 'submissions_by_call',
               'repository_requests', 'elapsed_seconds', 'started_epoch', 'finished_epoch', 'final_test_opened',
               'weights_public', 'eligible_for_final_promotion')}
    files = {'README.md': '\n'.join(lines), 'summary.json': json.dumps(summary, indent=2, allow_nan=False)+'\n',
             'submissions-by-call.svg': figure(value['submissions_by_call'])}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if (output.is_symlink() or {file.name for file in output.iterdir()} != set(files)
                or any((output / name).is_symlink() or (output / name).read_text(encoding='utf-8') != body
                       for name, body in files.items())):
            raise ValueError('Preserve the exact original diagnostic report')
    else:
        with TemporaryDirectory(prefix='.diagnostic-report-', dir=output.parent) as parent:
            staged = Path(parent) / 'results'
            staged.mkdir()
            for name, body in files.items():
                (staged / name).write_text(body, encoding='utf-8')
            staged.rename(output)
    return totals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('observations', 'registration', 'output'):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    print(json.dumps(render(json.loads(args.observations.read_bytes()), json.loads(args.registration.read_bytes()),
                            args.output), allow_nan=False))
