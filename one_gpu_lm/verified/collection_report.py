"""Report formal collection from numeric receipts, without reading private data."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from .cycle_report import hardware, write_csv
from .guard_audit import decision


def summarize(protocol, controller, audit, costs, integrity, supervisor, samples, export=None):
    expected = protocol['task_ids']
    if ([r['id'] for r in audit['rows']] != expected or [r['id'] for r in costs] != expected
            or len(set(expected)) != len(expected)):
        raise ValueError('Every registered task must appear once, in frozen order')
    if (audit['protocol_sha256'] != controller['protocol_sha256']
            or audit['protocol_sha256'] != integrity['protocol_sha256']):
        raise ValueError('Collection protocol receipt identity differs')
    if audit['decision'] != decision(protocol, audit['rows'], controller):
        raise ValueError('Recorded data gate differs from registered rules')
    completed = {r['id']: r for r in controller['tasks']}
    if len(completed) != len(controller['tasks']) or not set(completed) <= set(expected):
        raise ValueError('Duplicate or unexpected completed task')
    for row in costs:
        if row['id'] in completed:
            original = completed[row['id']]
            if any(row[k] != original[k] for k in ('attempted_calls', 'generated_tokens', 'complete')):
                raise ValueError('Completed-task cost differs from actual trajectory receipt')
    screened = {r['id']: r for r in audit['rows']}
    rows = []
    for cost in costs:
        task = screened[cost['id']]; original = completed.get(cost['id'], {})
        rows.append({**cost, 'repository': task['repository'], 'language': task['language'],
                     'correct': original.get('grade', {}).get('passed', False),
                     'submitted': original.get('submitted', False),
                     'eligible_for_sft': task['eligible_for_sft'],
                     'input_replay_passed': task['input_replay_passed'],
                     'encoding_error_count': len(task.get('encoding_errors', [])),
                     'positive_action_records': len(task.get('encoded_actions', [])) if task['eligible_for_sft'] else 0,
                     'excluded_unexecuted_requests': len(task.get('excluded_unexecuted_action_calls', [])),
                     'first_edit_call': original.get('first_edit_call'),
                     'terminal_reason': original.get('terminal_reason', 'incomplete' if cost['trace_present'] else 'not_run')})
    start, end = controller['started_epoch'], controller['finished_epoch']
    duration = end - start
    if duration < 0:
        raise ValueError('Invalid collection interval')
    status_checks = {
        'data_gate_passed': audit['decision']['passed'],
        'all_tasks_completed': controller['complete'] and len(completed) == len(expected),
        'all_generation_cost_accounted': all(r['trace_present'] and not r['unknown_inflight_generation_cost']
                                             and r['attempted_calls'] == r['completed_generation_receipts'] for r in rows),
        'supervisor_finished_normally': supervisor.get('complete') is True and not supervisor.get('timed_out')
                                         and supervisor.get('worker_returncode') == 0,
        'backend_stopped': controller.get('backend_stopped') is True,
        'within_original_supervised_time_limit': end <= controller['deadline_epoch']+protocol.get('supervisor_grace_seconds', 0),
        'frozen_files_unchanged': integrity.get('passed') is True and not integrity.get('mismatches'),
        'no_parameter_or_heldout_work': controller.get('training_updates') == 0
                                      and controller.get('development_tasks_scored') == 0
                                      and controller.get('final_test_opened') is False,
    }
    by_repo = []
    for repo in protocol['feasibility_gate']['repositories']:
        group = [r for r in rows if r['repository'] == repo]
        by_repo.append({'repository': repo, 'registered': len(group),
            'completed': sum(r['complete'] for r in group), 'correct': sum(r['correct'] for r in group),
            'eligible_full_episodes': sum(r['eligible_for_sft'] for r in group),
            'attempted_calls': sum(r['attempted_calls'] for r in group),
            'generated_tokens': sum(r['generated_tokens'] for r in group)})
    value = {'schema_version': 1, 'stage': 'formal_training_only_collection', 'protocol_sha256': audit['protocol_sha256'],
        'started_epoch': start, 'finished_epoch': end, 'pipeline_seconds': duration,
        'single_gpu_pipeline_hours': duration/3600,
        'started_utc': datetime.fromtimestamp(start, timezone.utc).isoformat(),
        'finished_utc': datetime.fromtimestamp(end, timezone.utc).isoformat(),
        'registered_maximum_pipeline_seconds': protocol['max_pipeline_seconds'],
        'deadline_epoch': controller['deadline_epoch'],
        'seconds_past_original_deadline': max(0, end-controller['deadline_epoch']),
        'supervisor_grace_seconds': protocol.get('supervisor_grace_seconds', 0),
        'tasks_registered': len(expected), 'tasks_completed': len(completed),
        'correct_tasks': sum(r['correct'] for r in rows),
        'eligible_full_episodes': sum(r['eligible_for_sft'] for r in rows),
        'eligible_action_records': sum(r['positive_action_records'] for r in rows),
        'supervised_tokens': sum(a['supervised_tokens'] for r in audit['rows'] if r['eligible_for_sft']
                                 for a in r.get('encoded_actions', [])),
        'known_cost': {k: sum(r[k] for r in rows) for k in ('attempted_calls', 'generated_tokens', 'input_tokens')},
        'unknown_inflight_tasks': [r['id'] for r in rows if r['unknown_inflight_generation_cost']],
        'not_run_tasks': [r['id'] for r in rows if not r['trace_present']],
        'partial_tasks': [r['id'] for r in rows if r['trace_present'] and not r['complete']],
        'unexecuted_requests_retained_in_cost': sum(r['excluded_unexecuted_requests'] for r in rows),
        'termination_counts': dict(Counter(r['terminal_reason'] for r in rows)),
        'decision': audit['decision'], 'checks': status_checks,
        'collection_verified_for_candidate_preparation': all(status_checks.values()),
        'private_export': export, 'repositories': by_repo, 'tasks': rows,
        'hardware': hardware(samples, controller), 'optimizer_updates': 0, 'new_development_scores': 0,
        'final_test_opened': False, 'model_quality_improvement_established': False,
        'weight_publication_allowed': False}
    return value


def report(directory, render=True):
    root = Path(directory)
    read = lambda name: json.loads((root/name).read_text())
    protocol, controller, audit = read('protocol.json'), read('controller.json'), read('audit.json')
    with (root/'telemetry.csv').open() as stream:
        samples = list(csv.DictReader(stream))
    export = read('export-summary.json') if (root/'export-summary.json').exists() else None
    value = summarize(protocol, controller, audit, read('task-costs.json'), read('integrity.json'),
                      read('supervisor.json'), samples, export)
    (root/'summary.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    write_csv(root/'tasks.csv', value['tasks'], list(value['tasks'][0]))
    write_csv(root/'repositories.csv', value['repositories'], list(value['repositories'][0]))
    if render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        cumulative = []; total = 0
        for row in value['tasks']:
            total += row['eligible_for_sft']; cumulative.append(total)
        axes[0].step(range(1, len(cumulative)+1), cumulative, where='post', color='#167d8d', label='Qualified full episodes')
        axes[0].axhline(protocol['feasibility_gate']['minimum_eligible_full_episodes'], color='#bd573d', ls='--', label='Frozen minimum')
        axes[0].set(xlabel='Registered task in fixed order', ylabel='Complete eligible episodes', title='Collection yield; not model improvement',
                    ylim=(0, max(1, total, protocol['feasibility_gate']['minimum_eligible_full_episodes'])*1.1))
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True)); axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].legend(fontsize=8)
        group = value['repositories']; x = list(range(len(group)))
        for offset, key, color, label in [(-.23, 'completed', '#989fa5', 'Completed'),
                (0, 'correct', '#c78b24', 'Correct repair'), (.23, 'eligible_full_episodes', '#167d8d', 'SFT eligible')]:
            axes[1].bar([i+offset for i in x], [r[key] for r in group], width=.23, color=color, label=label)
        axes[1].set(xticks=x, xticklabels=[r['repository'].split('/')[-1] for r in group], ylabel='Unique training tasks', title='Coverage after whole-episode screening')
        axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].tick_params(axis='x', rotation=25); axes[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(root/'collection-yield.png', dpi=170); plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        hours = [(datetime.fromisoformat(r['utc']).timestamp()-value['started_epoch'])/3600 for r in samples]
        axes[0].plot(hours, [float(r['utilization_percent']) for r in samples], lw=.7)
        axes[0].set(ylabel='GPU utilization (%)', ylim=(0, 105))
        axes[1].plot(hours, [float(r['memory_used_mib'])/1024 for r in samples], lw=.7, color='#167d8d')
        axes[1].set(xlabel='Hours since formal collection start', ylabel='GPU memory (GiB)')
        if not samples:
            for ax in axes: ax.text(.5, .5, 'No hardware samples', ha='center', transform=ax.transAxes)
        fig.tight_layout(); fig.savefig(root/'hardware.png', dpi=170); plt.close(fig)
    zone = timezone(timedelta(hours=8))
    start = datetime.fromtimestamp(value['started_epoch'], zone).isoformat()
    end = datetime.fromtimestamp(value['finished_epoch'], zone).isoformat()
    label = '通过' if value['collection_verified_for_candidate_preparation'] else '未通过'
    lines = ['# 025：正式训练轨迹采集结果', '', f'**采集与审计：{label}。** 本阶段没有更新学生参数，也没有新开发集评分。', '',
        f"固定 {value['tasks_registered']} 题、每题一次，完成 {value['tasks_completed']} 题，正确修复 {value['correct_tasks']} 题。整条轨迹通过执行取证和完整编码检查的有 {value['eligible_full_episodes']} 条，对应 {value['eligible_action_records']} 条动作记录；动作数不当作独立轨迹数。",
        f"北京时间 {start} 至 {end}，单卡流水线 {value['single_gpu_pipeline_hours']:.3f} 小时，包含模型加载、CPU 工具和判分。原时限 {protocol['max_pipeline_seconds']/3600:g} 小时，结束时间超过原截止 {value['seconds_past_original_deadline']:.2f} 秒；监督收尾宽限 {value['supervisor_grace_seconds']} 秒。", '',
        f"全部已知调用 {value['known_cost']['attempted_calls']:,} 次，生成 token {value['known_cost']['generated_tokens']:,}，输入 token {value['known_cost']['input_tokens']:,}。未知在途生成 {len(value['unknown_inflight_tasks'])} 题，未运行 {len(value['not_run_tasks'])} 题，部分运行 {len(value['partial_tasks'])} 题；未知成本不算零。", '',
        '| 检查 | 结果 |', '| --- | --- |']
    lines += [f"| {k} | {'通过' if v else '未通过'} |" for k, v in value['checks'].items()]
    lines += ['', '| 仓库 | 登记 | 完成 | 正确修复 | 合格完整轨迹 |', '| --- | --- | --- | --- | --- |']
    lines += [f"| {r['repository']} | {r['registered']} | {r['completed']} | {r['correct']} | {r['eligible_full_episodes']} |" for r in value['repositories']]
    lines += ['', '![轨迹产出及仓库覆盖](collection-yield.png)', '![硬件采样](hardware.png)', '',
        '逐题成本和失败见 [tasks.csv](tasks.csv)，完整编码及请求复核见 [audit.json](audit.json)，硬件采样覆盖、原截止与其他计数见 [summary.json](summary.json)。采集未运行新开发题，未重用此前试跑轨迹。',
        '合格轨迹只允许进入单独冻结的唯一候选训练准备，不证明参数收益。外部可复算公开数字并复用流程；原始语料、题目、生成轨迹和权重保持私有，不能据此原样重跑所有私有任务。公开上游源码可能已进入预训练，仓库隔离不等于预训练去污染。',
        '最终测试继续封存，不根据本轮训练数据产出发布 ModelScope 参数。', '',
        '复算：`python -m one_gpu_lm.verified.collection_report --results <本目录> --no-figures`。安装 matplotlib 后省略 `--no-figures` 可重绘曲线。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--results', type=Path, required=True); p.add_argument('--no-figures', action='store_true')
    a = p.parse_args(); value = report(a.results, render=not a.no_figures)
    print(json.dumps({'eligible_full_episodes': value['eligible_full_episodes'],
                      'collection_verified_for_candidate_preparation': value['collection_verified_for_candidate_preparation']}))
