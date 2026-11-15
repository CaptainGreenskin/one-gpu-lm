"""Recompute the registered development comparison from public numeric receipts.

No model, test runner, private prompt, patch or trajectory is needed. Figures are
descriptive; only factorial.analyze defines the preregistered decision.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from itertools import combinations
import json
import math
from pathlib import Path

from .factorial import ARMS, analyze, effects, schedule


ROW_FIELDS = ('id', 'repository', 'family', 'arm', 'repetition', 'run_id', 'complete',
              'audit_passed', 'success_within_budget', 'grader_passed', 'grade_reason',
              'submitted', 'terminal_reason', 'first_edit_call', 'invalid_actions',
              'rejected_unexecuted_requests', 'attempted_calls', 'generated_tokens',
              'input_tokens', 'seconds', 'actor_seconds', 'unknown_inflight_generation_cost')
CURVE_FIELDS = ('record', 'update', 'tokens', 'supervised_tokens', 'loss', 'seconds')
UPDATE_FIELDS = ('step', 'records', 'grad_norm', 'learning_rate', 'seconds')
HARDWARE_FIELDS = ('temperature_c', 'power_w', 'utilization_percent', 'memory_used_mib')
PHASE_FIELDS = ('started_epoch', 'finished_epoch', 'complete', 'optimizer_steps', 'action_records_consumed',
                'full_episodes', 'selected_records', 'input_tokens', 'supervised_tokens', 'training_seconds',
                'peak_allocated_bytes', 'peak_reserved_bytes', 'saved_reload_greedy_ids_equal',
                'tasks_completed', 'tasks_registered', 'eligible_full_episodes', 'eligible_action_records',
                'attempted_calls', 'generated_tokens')


def pick(value, names):
    return {k: value[k] for k in names if k in value}


def public_input(spec, tasks, audit, collection, candidate, evaluation, curve, updates, telemetry):
    """Allowlist receipt content before publication; never export raw trajectories.

    Callers must supply the already-public cycle specification and approved task
    identifiers. Unknown fields, commands, model text and process IDs are dropped.
    """
    safe_rows = []
    for row in audit['rows']:
        value = pick(row, ROW_FIELDS)
        value['request_replay'] = pick(row.get('request_replay', {}),
            ('action_fingerprint', 'delivered_feedback_fingerprints'))
        safe_rows.append(value)
    value = {'schema_version': 1, 'cycle_spec': spec,
             'tasks': [pick(t, ('id', 'repository', 'family')) for t in tasks],
             'evaluation_audit': {'passed': audit.get('passed') is True,
                'checks': {'frozen_inputs_unchanged': audit.get('checks', {}).get('frozen_inputs_unchanged') is True},
                'rows': safe_rows},
             'candidate_curve': [pick(r, CURVE_FIELDS) for r in curve],
             'candidate_updates': [pick(r, UPDATE_FIELDS) for r in updates],
             'telemetry': {name: [pick(r, ('utc', *HARDWARE_FIELDS)) for r in telemetry.get(name, [])]
                           for name in ('collection', 'candidate', 'evaluation')}}
    for name, phase in [('collection', collection), ('candidate', candidate), ('evaluation', evaluation)]:
        value[name] = pick(phase, PHASE_FIELDS)
        value[name]['telemetry'] = pick(phase.get('telemetry', {}), ('samples', 'errors', 'thread_stopped'))
        if 'backend_lifecycles' in phase:
            value[name]['backend_lifecycles'] = [pick(r, ('index', 'role', 'started_epoch', 'ready_epoch', 'stopped_epoch'))
                                                for r in phase['backend_lifecycles']]
    return value


def checked_score(row):
    return (row.get('complete') is True and row.get('audit_passed') is True
            and 'success_within_budget' in row and not row.get('unknown_inflight_generation_cost', False))


def category(row):
    if not row.get('complete'):
        return 'not_run' if not row.get('attempted_calls') else 'incomplete'
    if not checked_score(row):
        return 'unverified'
    if row['success_within_budget']:
        return 'success'
    if row.get('grade_reason') in {'protected_file_tampering', 'missing_source', 'unsafe_workspace'}:
        return 'invalid_submission'
    if row.get('grade_reason') == 'no_source_change' or row.get('first_edit_call') is None:
        return 'no_final_source_edit'
    if row.get('grader_passed'):
        return 'correct_patch_outside_budget'
    return 'wrong_edit'


def phase_cost(receipt):
    start, end = receipt.get('started_epoch'), receipt.get('finished_epoch')
    seconds = None if start is None or end is None else end - start
    if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
        raise ValueError('Invalid phase wall-clock interval')
    loads = receipt.get('backend_lifecycles')
    load_seconds = None if loads is None else sum(r['ready_epoch'] - r['started_epoch'] for r in loads if 'ready_epoch' in r)
    return {'pipeline_seconds': seconds,
            'started_utc': None if start is None else datetime.fromtimestamp(start, timezone.utc).isoformat(),
            'finished_utc': None if end is None else datetime.fromtimestamp(end, timezone.utc).isoformat(),
            'reserved_single_gpu_hours': None if seconds is None else seconds / 3600,
            'complete': receipt.get('complete', False), 'backend_loads': None if loads is None else len(loads),
            'completed_backend_load_seconds': load_seconds,
            'incomplete_backend_loads': None if loads is None else sum('ready_epoch' not in r for r in loads),
            'cost_definition': 'single-GPU pipeline wall time, including loading, CPU tools and grading; not kernel-active time'}


def hardware(samples, receipt):
    start, end = receipt.get('started_epoch'), receipt.get('finished_epoch')
    stamps = [datetime.fromisoformat(r['utc']).timestamp() for r in samples]
    if stamps != sorted(stamps):
        raise ValueError('Hardware timestamps are out of order')
    errors = receipt.get('telemetry', {}).get('errors')
    result = {'samples': len(samples), 'collection_errors': errors,
              'maximum_sample_gap_seconds': max((b-a for a, b in zip(stamps, stamps[1:])), default=None),
              'sample_span_seconds': stamps[-1]-stamps[0] if stamps else None,
              'boundary_uncovered_seconds': (max(0, stamps[0]-start)+max(0, end-stamps[-1]))
                  if stamps and start is not None and end is not None else None,
              'means_are_sample_means_not_utilization_weighted': True}
    for key in HARDWARE_FIELDS:
        values = [float(r[key]) for r in samples]
        if any(not math.isfinite(v) for v in values):
            raise ValueError('Nonfinite hardware sample')
        result[key] = {'mean': sum(values)/len(values) if values else None,
                       'peak': max(values) if values else None}
    return result


def summarize(bundle):
    spec, tasks, audit = bundle['cycle_spec'], bundle['tasks'], bundle['evaluation_audit']
    rows = audit['rows']
    # Missing per-run audit booleans may never be treated as verified scores.
    analysis_rows = [dict(r, audit_passed=checked_score(r)) for r in rows]
    result = analyze(analysis_rows, tasks, spec, audit.get('passed') is True,
                     audit.get('checks', {}).get('frozen_inputs_unchanged') is True)
    result.update(schema_version=1, scope='new-development-only verified self-distillation',
                  conclusion='achieved' if result['development_parameter_gate_passed'] else 'not_achieved',
                  final_test_opened=False, weight_publication_allowed=False)
    result['arms'] = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        verified = [r for r in selected if checked_score(r)]
        result['arms'][arm] = {
            'recorded_runs': len(selected), 'verified_runs': len(verified),
            'successes': sum(r['success_within_budget'] for r in verified),
            'explicit_submissions': sum(bool(r.get('submitted')) for r in verified),
            'categories': dict(Counter(category(r) for r in selected)),
            'invalid_actions': sum(r.get('invalid_actions', 0) for r in selected),
            'unexecuted_rejections': sum(r.get('rejected_unexecuted_requests', 0) for r in selected),
            'known_cost': {key: sum(r.get(key, 0) for r in selected)
                           for key in ('attempted_calls', 'input_tokens', 'generated_tokens', 'seconds')},
            'unknown_inflight_runs': sum(bool(r.get('unknown_inflight_generation_cost')) for r in selected)}
    paired = result.get('tasks', [])
    families = sorted({r['family'] for r in paired})
    result['families'] = []
    for family in families:
        group = [r for r in paired if r['family'] == family]
        scores = {arm: sum(r['scores'][arm] for r in group)/len(group) for arm in ARMS}
        result['families'].append({'family': family, 'paired_tasks': len(group), 'scores': scores,
                                  'effects': effects(scores),
                                  'parameter_gain_tasks': sum(r['effects']['parameter'] > 0 for r in group),
                                  'parameter_regression_tasks': sum(r['effects']['parameter'] < 0 for r in group)})
    result['instability'] = {}
    for arm in ARMS:
        detail = []
        for task in tasks:
            group = sorted((r for r in rows if r['id'] == task['id'] and r['arm'] == arm),
                           key=lambda r: r['repetition'])
            if len(group) != 3 or not all(checked_score(r) for r in group):
                continue
            fingerprints = [r.get('request_replay', {}).get('action_fingerprint') for r in group]
            feedback = [r.get('request_replay', {}).get('delivered_feedback_fingerprints') for r in group]
            detail.append({'id': task['id'],
                           'outcome_disagreement': len({r['success_within_budget'] for r in group}) > 1,
                           'action_disagreement': None if any(v is None for v in fingerprints) else len(set(fingerprints)) > 1,
                           'action_disagreeing_repeat_pairs': None if any(v is None for v in fingerprints)
                               else sum(a != b for a, b in combinations(fingerprints, 2)),
                           'feedback_disagreement': None if any(v is None for v in feedback)
                               else any(a != b for a, b in combinations(feedback, 2))})
        result['instability'][arm] = {
            'tasks_with_three_verified_repeats': len(detail),
            'outcome_disagreement_tasks': sum(r['outcome_disagreement'] for r in detail),
            'action_fingerprints_missing_tasks': sum(r['action_disagreement'] is None for r in detail),
            'action_disagreement_tasks': sum(r['action_disagreement'] is True for r in detail),
            'feedback_disagreement_tasks': sum(r['feedback_disagreement'] is True for r in detail),
            'tasks': detail}
    # A point is added only once a task has all four arms and all three repeats.
    # This is a descriptive accumulation plot, never a checkpoint-learning curve.
    by_id = {r['id']: r for r in paired}
    ordered = [p['task_id'] for p in schedule([t['id'] for t in tasks])
               if p['repetition'] == 2 and p['arm'] == 'base_current' and p['task_id'] in by_id]
    totals = dict.fromkeys(ARMS, 0.)
    result['outcome_curve'] = []
    for n, task_id in enumerate(ordered, 1):
        for arm in ARMS:
            totals[arm] += by_id[task_id]['scores'][arm]
        result['outcome_curve'].append({'paired_tasks': n, **{a: totals[a]/n for a in ARMS}})
    result['phases'] = {name: phase_cost(bundle[name]) for name in ('collection', 'candidate', 'evaluation')}
    result['hardware'] = {name: hardware(bundle.get('telemetry', {}).get(name, []), bundle[name])
                          for name in ('collection', 'candidate', 'evaluation')}
    result['training'] = {
        **pick(bundle['candidate'], ('complete', 'optimizer_steps', 'action_records_consumed', 'full_episodes',
            'selected_records', 'input_tokens', 'supervised_tokens', 'training_seconds',
            'peak_allocated_bytes', 'peak_reserved_bytes', 'saved_reload_greedy_ids_equal')),
        'curve': [pick(r, CURVE_FIELDS) for r in bundle.get('candidate_curve', [])],
        'updates': [pick(r, UPDATE_FIELDS) for r in bundle.get('candidate_updates', [])]}
    result['collection_counts'] = pick(bundle['collection'], ('tasks_completed', 'tasks_registered',
        'eligible_full_episodes', 'eligible_action_records', 'attempted_calls', 'generated_tokens', 'input_tokens'))
    result['limitations'] = spec['analysis'].get('limitations', '')
    result['private_task_specs_limit_external_exact_replication'] = True
    return result


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def figures(root, result, bundle):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#777777', '#c78b24', '#167d8d', '#6b57a6']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for arm, color in zip(ARMS, colors):
        rows = result['outcome_curve']
        axes[0].plot([r['paired_tasks'] for r in rows], [100*r[arm] for r in rows], label=arm, color=color)
    if not result['outcome_curve']:
        axes[0].text(.5, .5, 'No complete verified pairs', ha='center', transform=axes[0].transAxes)
    axes[0].set(xlabel='Complete unique tasks (all 3 repeats, 4 arms)', ylabel='Mean success (%)', ylim=(0, 105),
                title='Accumulation in frozen order; not learning')
    axes[0].legend(fontsize=8)
    for i, key in enumerate(('parameter', 'system', 'parameter_under_guard', 'interaction')):
        if key in result.get('effects', {}):
            mean = 100*result['effects'][key]; lo, hi = [100*v for v in result['intervals_95'][key]]
            axes[1].plot([lo, hi], [i, i], color='#167d8d', lw=3)
            axes[1].plot(mean, i, 'o', color='#142b42')
    axes[1].axvline(0, ls='--', color='#999999')
    axes[1].set(yticks=range(4), yticklabels=['Parameter (primary)', 'System', 'Parameter under guard', 'Interaction'],
                xlabel='Percentage points; paired cluster bootstrap 95% interval', title='Separate parameter and runtime effects')
    fig.tight_layout(); fig.savefig(root/'outcomes.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    curve = result['training']['curve']; updates = result['training']['updates']
    axes[0].plot([r['record'] for r in curve], [r['loss'] for r in curve], color='#167d8d', lw=.7,
                 marker='o' if len(curve) == 1 else None)
    axes[0].set(xlabel='Action record in fixed training order', ylabel='Assistant-token loss',
                title='Training loss on different examples; not a validation metric')
    axes[1].plot([r['step'] for r in updates], [r['grad_norm'] for r in updates], color='#6b57a6', lw=.8,
                 marker='o' if len(updates) == 1 else None)
    axes[1].set(xlabel='Optimizer update', ylabel='Gradient norm before clipping')
    fig.tight_layout(); fig.savefig(root/'training.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5))
    for col, phase in enumerate(('collection', 'candidate', 'evaluation')):
        samples = bundle.get('telemetry', {}).get(phase, [])
        start = bundle[phase].get('started_epoch')
        if samples and start is not None:
            hours = [(datetime.fromisoformat(r['utc']).timestamp()-start)/3600 for r in samples]
            axes[0, col].plot(hours, [float(r['utilization_percent']) for r in samples], lw=.7)
            axes[1, col].plot(hours, [float(r['memory_used_mib'])/1024 for r in samples], lw=.7, color='#167d8d')
        else:
            for row in range(2):
                axes[row, col].text(.5, .5, 'No samples', ha='center', transform=axes[row, col].transAxes)
        axes[0, col].set(title=phase, ylim=(0, 105), ylabel='Utilization (%)')
        axes[1, col].set(xlabel='Hours since phase start', ylabel='GPU memory (GiB)')
    fig.suptitle('Hardware samples include model loading and CPU tool work')
    fig.tight_layout(); fig.savefig(root/'hardware.png', dpi=170); plt.close(fig)


def report(directory, render=True):
    root = Path(directory)
    bundle = json.loads((root/'analysis-input.json').read_text())
    result = summarize(bundle)
    (root/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    rows = [{**pick(r, ROW_FIELDS), 'category': category(r)} for r in bundle['evaluation_audit']['rows']]
    write_csv(root/'runs.csv', rows, [*ROW_FIELDS, 'category'])
    pairs = [{'id': r['id'], 'repository': r['repository'], 'family': r['family'], **r['scores'], **r['effects']}
             for r in result.get('tasks', [])]
    write_csv(root/'pairs.csv', pairs, ['id', 'repository', 'family', *ARMS, 'parameter', 'system', 'parameter_under_guard', 'interaction'])
    families = [{'family': r['family'], 'paired_tasks': r['paired_tasks'], **r['scores'], **r['effects'],
                 'gain_tasks': r['parameter_gain_tasks'], 'regression_tasks': r['parameter_regression_tasks']}
                for r in result['families']]
    write_csv(root/'families.csv', families, ['family', 'paired_tasks', *ARMS, 'parameter', 'system', 'parameter_under_guard', 'interaction', 'gain_tasks', 'regression_tasks'])
    write_csv(root/'training-curve.csv', result['training']['curve'], CURVE_FIELDS)
    write_csv(root/'outcome-curve.csv', result['outcome_curve'], ['paired_tasks', *ARMS])
    if render:
        figures(root, result, bundle)
    lines = ['# 025：可验证自蒸馏四臂开发集结果', '',
        '**预登记参数提升门槛：'+('达成' if result['development_parameter_gate_passed'] else '未达成')+'。**', '',
        f"完整配对 {result['paired_tasks']}/{bundle['cycle_spec']['development']['unique_tasks']} 道独立任务；每题三次重复。重复运行不算额外独立任务。",
        '参数收益以 student_current − base_current 为主指标；系统收益与交互单列。缺失或未核验配对不能支持达标，普通失败计入分母，全部已知消耗保留。', '',
        '| 组别 | 完整配对平均成功率 | 核验通过的运行数 | 已知调用 | 已知生成 token |',
        '| --- | --- | --- | --- | --- |']
    for arm in ARMS:
        value = result['arms'][arm]; score = result.get('scores', {}).get(arm)
        display = '无完整配对' if score is None else f'{score*100:.2f}%'
        lines.append(f"| {arm} | {display} | {value['verified_runs']} | {value['known_cost']['attempted_calls']} | {value['known_cost']['generated_tokens']} |")
    lines += ['', '| 效应 | 百分点 | 95% 配对分层区间 |', '| --- | --- | --- |']
    for key, value in result.get('effects', {}).items():
        lo, hi = result['intervals_95'][key]
        lines.append(f'| {key} | {100*value:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
    lines += ['', '| 验收项 | 结果 |', '| --- | --- |']
    lines += [f"| {key} | {'通过' if value else '未通过'} |" for key, value in result.get('gate_checks', {}).items()]
    lines += ['', '| 阶段 | 单卡流水线小时 | 后端加载次数 | 已完成加载秒数 |', '| --- | --- | --- | --- |']
    for phase, value in result['phases'].items():
        hours = value['reserved_single_gpu_hours']
        display = '未完整记录' if hours is None else f'{hours:.3f}'
        loads = '未单独记录' if value['backend_loads'] is None else str(value['backend_loads'])
        seconds = value['completed_backend_load_seconds']
        load_time = '未单独记录' if seconds is None else f'{seconds:.2f}'
        lines.append(f'| {phase} | {display} | {loads} | {load_time} |')
    lines += ['', '卡时口径为本轮正式采集、唯一候选训练和四臂评测各自的控制器起止时间，包含加载、CPU 工具和判分，不等于 GPU 核函数活跃时间。此前试跑、校准和容量检查的成本保留在各自报告，不在此表中重复累计。未完成或未单独计时的后端加载不会被算作零成本，其耗时仍在流水线内；未知在途生成成本单独记录。',
        '训练损失来自依次消费的不同动作样本，不能当作验证能力曲线。结果累计曲线仅按固定顺序加入完整任务，不用于选择 checkpoint。', '',
        '![四臂结果与效应区间](outcomes.png)', '![训练损失和梯度](training.png)', '![三阶段硬件记录](hardware.png)', '',
        '逐题、逐仓库配对见 [pairs.csv](pairs.csv) 和 [families.csv](families.csv)；不修改代码、错误修改、非法提交、超预算以及未完成/未核验状态见 [runs.csv](runs.csv)。动作和结果的重复波动、硬件采样覆盖及所有已知成本见 [summary.json](summary.json)。', '',
        '局限：仅六个仓库家族上的短构造故障，不能据此声称通用编码、长任务或先进小模型水平。公开源码可能进入底座预训练；仓库隔离不等于预训练去污染。任务定义保持私有，外部可重算公开统计、复用流程，但无法原样重跑全部私有任务。',
        '本轮属于同底座高预算轨迹的自蒸馏。既有最终测试仍封存，开发结果不授权发布 ModelScope 参数。', '',
        '统计复算：`python -m one_gpu_lm.verified.cycle_report --results <本目录>`；安装 matplotlib 后生成图，`--no-figures` 可仅用标准库复算表格与结论。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    value = report(args.results, render=not args.no_figures)
    print(json.dumps({'conclusion': value['conclusion'], 'paired_tasks': value['paired_tasks'],
                      'final_test_opened': False, 'weight_publication_allowed': False}))
