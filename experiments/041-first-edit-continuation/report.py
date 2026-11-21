"""Project 041 receipts onto an anonymous paired table, then render its curves."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.verified.recovery_report import hardware
from one_gpu_lm.verified.tasks import relative


ARMS = ('direct', 'low')
PRIVATE_ARMS = dict(zip(ARMS, ('control', 'correction')))
LANGUAGES = {'python', 'javascript'}
FLAGS = ('correct_patch', 'verified_and_submitted_within_budget', 'qualified', 'timeout_recovered')
COUNTS = ('edit_actions', 'attempted_calls', 'generated_tokens', 'input_tokens', 'invalid_actions')
QUALIFICATION_REASONS = {
    'repair_or_submission_failed', 'protected_file_changed', 'first_edit_not_the_verified_final_patch',
    'final_verification_chain_missing', 'historical_visible_failure_dropped',
    'untruncated_student_target_does_not_fit', 'student_encoding_rejected',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(valid, message):
    if not valid:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def count(value):
    return type(value) is int and value >= 0


def project_cell(cell, budget):
    if cell['audited'] is False:
        require(cell['reason'] in {'attempt_not_completed', 'audit_failed'}, 'Unregistered unaudited disposition')
        return {'audited': False, 'reason': cell['reason']}
    require(cell['audited'] is True and all(type(cell[k]) is bool for k in FLAGS)
            and all(count(cell[k]) for k in COUNTS) and number(cell['seconds']),
            'Unexpected paired measurement type')
    calls = cell['attempted_calls']
    edit = cell['first_edit_call']
    require(0 < calls <= budget['calls'] and cell['edit_actions'] <= calls
            and cell['invalid_actions'] <= calls and cell['generated_tokens'] <= budget['tokens']
            and 0 < cell['input_tokens'] <= calls * budget['input']
            and ((type(edit) is int and 1 <= edit <= calls) if cell['edit_actions'] else edit is None),
            'Recorded cost or first edit differs from paired budget')
    success = cell['verified_and_submitted_within_budget']
    reason = cell['qualification_reason']
    require((not success or cell['correct_patch'] and cell['seconds'] <= budget['seconds']
                and not cell['timeout_recovered'])
            and ((reason is None and success and cell['edit_actions'] == 1) if cell['qualified']
                 else reason in QUALIFICATION_REASONS), 'Inconsistent qualification outcome')
    return {key: cell[key] for key in ('audited', 'first_edit_call', 'seconds', 'qualification_reason', *FLAGS, *COUNTS)}


def project_pairs(pairs, design, raw=False):
    cap = design['total_budget']
    budget = {'calls': cap['max_model_calls'] - design['historical_calls'], 'tokens': cap['max_total_generated_tokens'],
              'input': cap['max_input_tokens'], 'seconds': cap['max_task_seconds']}
    require(len(pairs) == design['task_count'], 'Require all registered pairs, including incomplete pairs')
    families = {}
    result = []
    for i, pair in enumerate(pairs, 1):
        require(pair['pair'] == i and pair['language'] in LANGUAGES, 'Unregistered pair number or language')
        family = pair['family']
        if raw:
            family = families.setdefault(family, len(families) + 1)
        else:
            require(type(family) is int and 1 <= family <= len(pairs), 'Public family must be an anonymous number')
        arms = PRIVATE_ARMS if raw else {arm: arm for arm in ARMS}
        require(set(pair['arms']) == set(arms.values()), 'Both arm dispositions are required')
        result.append({'pair': i, 'family': family, 'language': pair['language'],
                       'arms': {arm: project_cell(pair['arms'][name], budget) for arm, name in arms.items()}})
    return result


def summarize(pairs, design, receipt):
    paired = [p for p in pairs if all(p['arms'][a]['audited'] for a in ARMS)]
    qualified = [p for p in pairs if any(p['arms'][a].get('qualified') for a in ARMS)]
    families = len({p['family'] for p in qualified})
    calls = sum(cell.get('attempted_calls', 0) for p in pairs for cell in p['arms'].values())
    require(receipt['journal_requests_verified'] == calls and receipt['qualified_tasks'] == len(qualified)
            and receipt['qualified_families'] == families and (not receipt['audit_passed'] or len(paired) == len(pairs)),
            'Audited counts differ from the retained paired evidence')
    gate = (receipt['audit_passed'] and len(qualified) >= design['minimum_qualified_tasks']
            and families >= design['minimum_qualified_families'])
    require(gate == receipt['collection_gate_passed'], 'Registered collection gate differs')
    export = receipt['private_export_counts']
    expected = {'records', 'supervised_tokens'}
    require((export is None and not gate) or (gate and set(export) == expected
            and export['records'] == 3 * len(qualified) and count(export['supervised_tokens'])
            and export['supervised_tokens'] >= export['records']), 'Private yield does not match the collection gate')
    by_arm = {arm: [p['arms'][arm] for p in paired] for arm in ARMS}
    outcomes = {}
    for arm, rows in by_arm.items():
        outcomes[arm] = {**{key: sum(r[key] for r in rows) for key in (*FLAGS, *COUNTS, 'seconds')},
                         'first_edited': sum(r['first_edit_call'] is not None for r in rows)}
    contrasts = {}
    for key in ('correct_patch', 'verified_and_submitted_within_budget', 'qualified'):
        deltas = [p['arms']['low'][key] - p['arms']['direct'][key] for p in paired]
        contrasts[key] = {'gain': deltas.count(1), 'loss': deltas.count(-1), 'net': sum(deltas)}
    return {'planned_pairs': len(pairs), 'complete_audited_pairs': len(paired),
            'audited_arms': sum(cell['audited'] for p in pairs for cell in p['arms'].values()),
            'verified_requests': calls, 'outcomes': outcomes, 'paired_low_minus_direct': contrasts,
            'qualified_tasks': len(qualified), 'qualified_families': families,
            'collection_gate_passed': gate, 'private_export_counts': export}


def capture(root, protocol_path):
    """Run on the workstation. Return only allowlisted numeric data and anonymous IDs."""
    root, protocol_path = Path(root), Path(protocol_path)
    plan = read(protocol_path)
    out = root / relative(plan['run_directory'])
    design_path = root / relative(plan['design_path'])
    design = read(design_path)
    controller_path = out / 'controller.json'
    audit_path = out / 'audit.json'
    controller, audit = read(controller_path), read(audit_path)
    identity = sha256(protocol_path)
    require(controller['backend_stopped'] and controller['protocol_sha256'] == audit['protocol_sha256'] == identity
            and not controller['final_test_opened'] and not audit['final_test_opened']
            and controller['optimizer_updates'] == audit['new_optimizer_updates'] == 0
            and type(audit['passed']) is bool and type(audit['collection_gate_passed']) is bool,
            'Require a terminal audited training-only run')
    terminal = controller['stage']
    require(terminal in {'completed', 'failed'} and bool(controller['complete']) == (terminal == 'completed')
            and (not audit['passed'] or controller['complete'] and controller['frozen_files_unchanged']),
            'Controller terminal stage differs from the audit')
    started, finished = (controller[k] for k in ('started_epoch', 'finished_epoch'))
    require(number(started) and number(finished) and finished > started, 'Invalid execution times')
    export = audit['export']
    if export is not None:
        private = out / 'private' / 'first-edit-export'
        require(read(private / 'manifest.json') == export and export['raw_data_public'] is False
                and sha256(private / 'data.jsonl') == export['data_sha256'], 'Private training export bytes differ')
    receipt = {'schema_version': 1, 'protocol_sha256': identity, 'design_sha256': sha256(design_path),
               'controller_sha256': sha256(controller_path), 'audit_sha256': sha256(audit_path),
               'started_epoch': started, 'finished_epoch': finished, 'terminal_stage': terminal,
               'audit_passed': audit['passed'], 'collection_gate_passed': audit['collection_gate_passed'],
               'qualified_tasks': audit['qualified_tasks'], 'qualified_families': audit['qualified_families'],
               'journal_requests_verified': audit['journal_requests_verified'],
               'private_export_counts': None if export is None else {k: export[k] for k in ('records', 'supervised_tokens')},
               'pairs': project_pairs(audit['pairs'], design, raw=True),
               'new_optimizer_updates': 0, 'final_test_opened': False}
    samples = hardware(out / 'telemetry.csv', {k: controller[k] for k in ('telemetry', 'started_epoch', 'finished_epoch')})
    # This statistics helper carries the collector's object through. Publish only its fixed public fields.
    samples['collector'] = {k: samples['collector'][k] for k in ('samples', 'errors', 'thread_stopped')}
    require(samples['samples'] == samples['collector']['samples'], 'Telemetry sample inventory differs')
    receipt['hardware'] = samples
    summarize(receipt['pairs'], design, receipt)
    return receipt


def figure(path, pairs, new_calls):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained', sharey=True)
    x = list(range(new_calls + 1))
    for arm, color, label in [('direct', '#225FA6', 'Thinking disabled'), ('low', '#B74725', 'Low reasoning')]:
        rows = [p['arms'][arm] for p in pairs]
        axes[0].step(x, [sum(r['first_edit_call'] is not None and r['first_edit_call'] <= j for r in rows) for j in x],
                     where='post', color=color, label=label, linewidth=2)
        axes[1].step(x, [sum(r['verified_and_submitted_within_budget'] and r['attempted_calls'] <= j for r in rows) for j in x],
                     where='post', color=color, label=label, linewidth=2)
    for ax, title in zip(axes, ('First source edit', 'Verified and submitted')):
        ax.set(title=title, xlabel='New model calls', xlim=(0, new_calls), ylim=(0, max(1, len(pairs)) + .25), xticks=x)
        ax.set_yticks(list(range(len(pairs) + 1)))
        ax.grid(alpha=.22)
        ax.legend(loc='lower right')
    axes[0].set_ylabel('Cumulative tasks')
    fig.savefig(path, dpi=160, metadata={'Software': 'one-gpu-lm'})
    plt.close(fig)


def render(directory, design_path):
    directory = Path(directory)
    receipt, design = read(directory / 'measurements.json'), read(design_path)
    hashes = ('protocol_sha256', 'design_sha256', 'controller_sha256', 'audit_sha256')
    require(all(re.fullmatch(r'[0-9a-f]{64}', receipt[k]) for k in hashes)
            and receipt['design_sha256'] == sha256(design_path) and not receipt['final_test_opened']
            and receipt['new_optimizer_updates'] == 0, 'Public result identity or scope differs')
    pairs = project_pairs(receipt['pairs'], design)
    require(pairs == receipt['pairs'], 'Unexpected fields in the anonymous pair table')
    summary = summarize(pairs, design, receipt)
    write_json(directory / 'summary.json', summary)
    write_json(directory / 'hardware.json', receipt['hardware'])
    with (directory / 'pairs.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ('audited', 'first_edit_call', 'seconds', *FLAGS, *COUNTS)
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['pair', 'family', 'language', 'arm', *fields])
        for pair in pairs:
            for arm in ARMS:
                writer.writerow([pair['pair'], pair['family'], pair['language'], arm,
                                 *[pair['arms'][arm].get(k) for k in fields]])
    paired = [p for p in pairs if all(c['audited'] for c in p['arms'].values())]
    cap = design['total_budget']['max_model_calls'] - design['historical_calls']
    figure(directory / 'first-edit.png', paired, cap)
    zone = timezone(timedelta(hours=8))
    start, finish = (datetime.fromtimestamp(receipt[key], zone).strftime('%Y-%m-%d %H:%M:%S')
                     for key in ('started_epoch', 'finished_epoch'))
    left, right = (summary['outcomes'][arm] for arm in ARMS)
    fields = [('首次实际改码', 'first_edited'), ('独立评分通过', 'correct_patch'),
              ('预算内验证并提交', 'verified_and_submitted_within_budget'), ('合格示范', 'qualified'),
              ('实际新调用', 'attempted_calls'), ('生成 token，含思考', 'generated_tokens'),
              ('输入 token', 'input_tokens'), ('代理累计秒数', 'seconds'), ('已取回超时回执', 'timeout_recovered')]
    table = '\n'.join(f'| {label} | {left[key]:g} | {right[key]:g} |' for label, key in fields)
    edge = summary['paired_low_minus_direct']['qualified']
    passed = '通过' if summary['collection_gate_passed'] else '未通过'
    (directory / 'README.md').write_text(f'''# 041 · 首次改码接续的配对结果

北京时间 {start} 至 {finish}。预注册 {summary['planned_pairs']} 对，完整审计 {len(paired)} 对。全体审计通过的分支共 {summary['audited_arms']} 个，核对了 {summary['verified_requests']} 条生成回执。

| 完整配对内指标 | 关闭思考 | 低档思考 |
|---|---:|---:|
{table}

低档思考组在合格示范上新增 {edge['gain']} 题、减少 {edge['loss']} 题，净变化 {edge['net']:+d} 题。合格示范覆盖 **{summary['qualified_tasks']} 道题、{summary['qualified_families']} 个家族**，采集门槛**{passed}**。

![新调用次数、首次改码与验证提交曲线](first-edit.png)

曲线和两组对照使用完整审计的配对。表中时间为代理累计耗时，历史前缀的成本另计。全部注册配对和未完整审计的分支保留在匿名表中。选题来自已观察的训练失败，结果用于判断能否采集首次改码示范。开发及最终验收另行执行。

[匿名配对 CSV](pairs.csv) · [聚合指标](summary.json) · [硬件统计](hardware.json) · [测量回执](measurements.json) · [冻结设计](../../design.json)
''', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('capture', 'render'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--design', type=Path)
    args = parser.parse_args()
    if args.action == 'capture':
        if args.root is None or args.protocol is None:
            parser.error('capture requires --root and --protocol')
        print(json.dumps(capture(args.root, args.protocol), allow_nan=False))
    else:
        if args.results is None or args.design is None:
            parser.error('render requires --results and --design')
        print(json.dumps(render(args.results, args.design), allow_nan=False))
