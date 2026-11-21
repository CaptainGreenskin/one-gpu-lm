"""Verify terminal 042 receipts and reproduce the training-only figures."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
from statistics import fmean

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.verified.recovery_report import hardware
from one_gpu_lm.verified.tasks import relative


RECEIPTS = ('result.json', 'curve.json', 'updates.json', 'selection.json', 'adapter.json', 'environment.json')
UPDATE_FIELDS = ('step', 'records', 'input_tokens', 'supervised_tokens', 'mean_loss', 'grad_norm', 'seconds')
TRAINING_CHECKS = ('complete', 'frozen_files_unchanged', 'all_selected_records_consumed',
                   'saved_reload_greedy_ids_equal', 'frozen_sample_unchanged', 'frozen_gradients_absent',
                   'adapter_changed')


def read(path):
    return json.loads(Path(path).read_bytes())


def require(valid, message):
    if not valid:
        raise ValueError(message)


def metric(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def positive_count(value):
    return type(value) is int and value > 0


def update_metrics(curve, updates):
    """Project only journaled optimizer windows into aggregate metrics."""
    projected = []
    cursor = 0
    for step, update in enumerate(updates, 1):
        records = update['records']
        require(update['step'] == step and positive_count(records), 'Invalid optimizer step or window length')
        window = curve[cursor:cursor + records]
        require(len(window) == records and all(
            row['record'] == index and row['update'] == step and metric(row['loss'])
            and positive_count(row['tokens']) and positive_count(row['supervised_tokens'])
            and row['supervised_tokens'] <= row['tokens']
            for index, row in enumerate(window, cursor + 1)), 'A completed optimizer window lacks its recorded examples')
        require(metric(update['grad_norm']) and metric(update['seconds']), 'Nonfinite optimizer metric')
        projected.append({'step': step, 'records': records, 'input_tokens': sum(row['tokens'] for row in window),
                          'supervised_tokens': sum(row['supervised_tokens'] for row in window),
                          'mean_loss': fmean(row['loss'] for row in window),
                          'grad_norm': update['grad_norm'], 'seconds': update['seconds']})
        cursor += records
    return projected, len(curve) - cursor


def capture(root, protocol_path, job_path, publication_path):
    plan, job, publication = map(read, (protocol_path, job_path, publication_path))
    identity = sha256(protocol_path)
    require(plan['candidate'] == '042-first-edit-sft' and publication['protocol_sha256'] == identity
            and re.fullmatch(r'[0-9a-f]{40}', publication['commit'])
            and publication['anonymous_files_verified'] is True and publication['ci_conclusion'] == 'success'
            and not plan['final_test_opened'] and not plan['weights_public'] and not job['final_test_opened']
            and job['phase'] in ('audited', 'failed'), 'Require a terminal run under the verified training registration')
    out = root / relative(plan['run_directory'])
    state_path = out / 'result.json'
    state = read(state_path) if state_path.is_file() else None
    audit_path = out / 'integrity.json'
    passed = job['phase'] == 'audited'
    if state is None:
        require(not passed and not out.exists(), 'Missing authoritative training result')
        start, end = (datetime.fromisoformat(job[k]).timestamp() for k in ('started_utc', 'finished_utc'))
    else:
        require(state['protocol_sha256'] == identity and state['stage'] in ('completed', 'failed', 'inputs_changed')
                and not state['weights_public'] and not state['final_test_opened'] and state['development_scores'] == 0,
                'Result differs from the registered training-only run')
        start, end = (state[k] for k in ('started_epoch', 'finished_epoch'))
    require(metric(start) and metric(end) and end > start, 'Invalid training time interval')

    integrity = None
    if passed:
        audit = read(audit_path)
        integrity = sha256(audit_path)
        require(job['adapter_integrity_sha256'] == integrity and job['protocol_sha256'] == identity
                and audit['passed'] is True and audit['protocol_sha256'] == identity and not audit['final_test_opened']
                and not audit['weights_public'] and audit['audit_model_generations'] == audit['audit_optimizer_updates'] == 0
                and state['stage'] == 'completed' and all(state[key] is True for key in TRAINING_CHECKS)
                and audit['optimizer_updates'] == plan['expected_optimizer_updates']
                and set(audit['receipt_sha256']) == set(RECEIPTS)
                and all(sha256(out / name) == audit['receipt_sha256'][name] for name in RECEIPTS),
                'Published measurements must match the exact completed training audit')

    update_file, curve_file = out / 'updates.json', out / 'curve.json'
    updates, tail = update_metrics(read(curve_file) if curve_file.is_file() else [],
                                   read(update_file) if update_file.is_file() else [])
    if state is not None:
        require(len(updates) == state['optimizer_steps']
                and sum(row['records'] for row in updates) == state['action_records_consumed'],
                'Optimizer receipts differ from the terminal training record')
    if passed:
        require(tail == 0 and len(updates) == plan['expected_optimizer_updates']
                and sum(row['records'] for row in updates) == plan['expected_records']
                and sum(row['input_tokens'] for row in updates) == state['input_tokens']
                and sum(row['supervised_tokens'] for row in updates) == state['supervised_tokens'],
                'Completed token or example totals differ from the audit')
    error = (state or {}).get('error_type') or job.get('error_type')
    if error is not None:
        require(re.fullmatch(r'[A-Z][A-Za-z]{0,59}', error), 'Expected an exception class without diagnostic text')
    telemetry = None
    if state is not None and (out / 'telemetry.csv').is_file():
        telemetry = hardware(out / 'telemetry.csv', state)
        telemetry['collector'] = {key: state['telemetry'][key] for key in ('samples', 'errors', 'thread_stopped')}
        require(telemetry['samples'] == telemetry['collector']['samples'], 'Hardware receipt sample count differs')
    value = {'schema_version': 1, 'protocol_sha256': identity, 'registration_commit': publication['commit'],
            'integrity_sha256': integrity, 'audit_passed': passed, 'stage': state['stage'] if state else 'not_started',
            'error_type': error, 'started_epoch': start, 'finished_epoch': end,
            'training_seconds': (state or {}).get('training_seconds'),
            'trainable_parameters': (state or {}).get('trainable_parameters'),
            'peak_allocated_bytes': (state or {}).get('peak_allocated_bytes'),
            'peak_reserved_bytes': (state or {}).get('peak_reserved_bytes'),
            'saved_reload_greedy_ids_equal': (state or {}).get('saved_reload_greedy_ids_equal'),
            'partially_processed_records': tail, 'updates': updates, 'hardware': telemetry,
            'final_test_opened': False, 'weights_public': False}
    require(all(value[name] is None or metric(value[name]) for name in
                ('training_seconds', 'trainable_parameters', 'peak_allocated_bytes', 'peak_reserved_bytes'))
            and (value['saved_reload_greedy_ids_equal'] is None or type(value['saved_reload_greedy_ids_equal']) is bool),
            'Unexpected metric type in the terminal training record')
    return value


def figure(path, updates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6), sharex=True, layout='constrained')
    steps = [row['step'] for row in updates]
    for ax, key, label, color in zip(axes, ('mean_loss', 'grad_norm'),
                                    ('Mean training loss', 'Gradient norm'), ('#225FA6', '#B74725')):
        ax.plot(steps, [row[key] for row in updates], color=color, linewidth=1.2, marker='.', markersize=3)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.23)
    axes[-1].set_xlabel('Optimizer update')
    fig.savefig(path, dpi=160, metadata={'Software': 'one-gpu-lm'})
    plt.close(fig)


def render(directory, registration_path):
    value = read(directory / 'measurements.json')
    registration = read(registration_path)
    planned = registration['training']
    require(value['protocol_sha256'] == registration['private_protocol_sha256']
            and not value['final_test_opened'] and not value['weights_public'],
            'Results differ from the published training registration')
    rows = value['updates']
    require(all(set(row) == set(UPDATE_FIELDS) and row['step'] == index
                and all(metric(row[key]) for key in UPDATE_FIELDS)
                for index, row in enumerate(rows, 1)), 'Unexpected public optimizer metrics')
    consumed = sum(row['records'] for row in rows)
    input_tokens = sum(row['input_tokens'] for row in rows)
    supervised = sum(row['supervised_tokens'] for row in rows)
    if value['audit_passed']:
        require(value['integrity_sha256'] and value['stage'] == 'completed'
                and value['saved_reload_greedy_ids_equal'] is True and value['partially_processed_records'] == 0
                and len(rows) == planned['optimizer_updates'] and consumed == planned['action_records']
                and input_tokens == planned['input_tokens'] and supervised == planned['supervised_tokens']
                and value['trainable_parameters'] == planned['trainable_parameters'],
                'Audited output does not match the complete planned optimizer run')
    with (directory / 'updates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, UPDATE_FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        figure(directory / 'training.png', rows)
    zone = timezone(timedelta(hours=8))
    start, end = (datetime.fromtimestamp(value[key], zone) for key in ('started_epoch', 'finished_epoch'))
    duration = (end - start).total_seconds()
    scope = '运行窗口' if value['stage'] == 'not_started' else '单卡训练流水线'
    outcome = '通过' if value['audit_passed'] else '未通过'
    lines = ['# 042 · 首次改码示范微调结果', '',
             f'北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}。{scope}用时 {duration:.2f} 秒。训练产物审计**{outcome}**。',
             '',
             f'完成 {len(rows)}/{planned["optimizer_updates"]} 次优化器更新，已计入 {consumed:,} 条动作、{input_tokens:,} 个输入 token 和 {supervised:,} 个监督 token。']
    if value['error_type']:
        lines += ['', f'终止时记录的异常类型为 `{value["error_type"]}`。仅统计已有优化器回执的窗口。']
    if value['partially_processed_records']:
        lines += ['', f'另有 {value["partially_processed_records"]} 条记录属于未完成的窗口，单独计数。']
    if value['training_seconds'] is not None:
        lines += ['', f'训练循环用时 {value["training_seconds"]:.2f} 秒。']
    if value['saved_reload_greedy_ids_equal'] is True:
        lines += ['', '保存后重载的贪心生成 ID 一致。']
    if value['hardware'] is not None:
        gpu = value['hardware']
        lines += ['', f'采集 {gpu["samples"]} 条 GPU 样本，采样失败 {gpu["collector"]["errors"]} 次。']
        if gpu['samples']:
            lines[-1] += (f'采样最高显存占用 {gpu["memory_used_mib"]["maximum"]:,.0f} MiB，最高温度 '
                          f'{gpu["temperature_c"]["maximum"]:.0f} °C，最长采样间隔 {gpu["maximum_gap_seconds"]:.2f} 秒。')
    if rows:
        lines += ['', '![训练损失和梯度范数](training.png)', '',
                  '每个损失点是该次更新内各动作损失的均值。编码修复成功率由独立开发对照另行评估。']
    lines += ['', '[更新指标](updates.csv) · [测量和硬件回执](measurements.json) · [训练登记](../../registration.json)']
    (directory / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'audit_passed': value['audit_passed'], 'optimizer_updates': len(rows), 'records': consumed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    capture_args = actions.add_parser('capture')
    for name in ('root', 'protocol', 'job', 'publication'):
        capture_args.add_argument('--' + name, type=Path, required=True)
    render_args = actions.add_parser('render')
    render_args.add_argument('--directory', type=Path, required=True)
    render_args.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    result = (capture(args.root, args.protocol, args.job, args.publication) if args.action == 'capture'
              else render(args.directory, args.registration))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
