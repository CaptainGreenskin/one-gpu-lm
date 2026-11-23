"""Reproduce audited 051 optimizer and hardware summaries from anonymous records."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
SHARED = runpy.run_path(str(HERE.parent / '042-first-edit-sft' / 'report.py'))
CHECKS = ('saved_reload_greedy_ids_equal', 'frozen_sample_unchanged', 'frozen_gradients_absent',
          'all_selected_records_consumed', 'frozen_files_unchanged')


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n')


def render(value, output):
    design, registration = (read(HERE / name) for name in ('design.json', 'registration.json'))
    summary = registration['selection']
    metrics = value['metrics']
    updates, tail = SHARED['update_metrics'](value['records'], value['updates'])
    counts = Counter(row['record_group_kind'] for row in value['records'])
    actual = {'input_tokens': sum(row['input_tokens'] for row in updates),
              'supervised_tokens': sum(row['supervised_tokens'] for row in updates),
              'target_counts': dict(counts)}
    if (value['protocol_sha256'] != registration['protocol_sha256'] or not value['integrity_sha256']
            or value['final_test_opened'] or value['weights_public'] or tail
            or len(updates) != metrics['optimizer_steps'] or len(updates) != summary['optimizer_updates']
            or sum(row['records'] for row in updates) != metrics['action_records_consumed']
            or metrics['action_records_consumed'] != summary['records']
            or any(summary[key] != item for key, item in actual.items())
            or any(metrics[key] != actual[key] for key in ('input_tokens', 'supervised_tokens'))
            or not all(metrics[key] is True for key in CHECKS)
            or metrics['trainable_parameters'] != design['trainable_parameters']
            or not metrics['started_epoch'] < metrics['finished_epoch']):
        raise ValueError('Only the complete audited training attempt may be summarized')
    gpu = value['hardware']
    if gpu['samples'] != metrics['telemetry']['samples'] or gpu['collector'] != metrics['telemetry']:
        raise ValueError('Hardware sample coverage differs from the training journal')
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'measurements.json', value)
    with (output / 'updates.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=updates[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(updates)
    SHARED['figure'](output / 'training.png', updates)
    seconds = metrics['finished_epoch'] - metrics['started_epoch']
    report = {**actual, 'updates': len(updates), 'records': summary['records'], 'episodes': summary['episodes'],
              'audit_passed': True, 'reload_generation_matches': True,
              'trainable_parameters': metrics['trainable_parameters'], 'pipeline_seconds': seconds,
              'training_seconds': metrics['training_seconds'], 'peak_allocated_bytes': metrics['peak_allocated_bytes'],
              'hardware': gpu, 'final_test_opened': False, 'weights_public': False}
    write(output / 'summary.json', report)
    zone = timezone(timedelta(hours=8))
    start, end = (datetime.fromtimestamp(metrics[key], zone) for key in ('started_epoch', 'finished_epoch'))
    lines = ['# 051 · 验证修复动作微调结果', '',
             f'完成 **{len(updates)}** 次更新，消耗 **{summary["records"]}** 条动作和 '
             f'**{summary["supervised_tokens"]:,}** 个监督 token。LoRA 可训练参数 {metrics["trainable_parameters"]:,}，'
             '底座保持冻结，保存后重新加载的贪心输出一致。训练产物通过独立审计。', '',
             f'北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}，'
             f'单卡流水线用时 {seconds:.0f} 秒，训练循环用时 {metrics["training_seconds"]:.0f} 秒。']
    if gpu['samples']:
        lines += ['', f'GPU 采样 {gpu["samples"]} 条，失败 {gpu["collector"]["errors"]} 次。'
                  f'平均利用率 {gpu["utilization_percent"]["mean"]:.1f}%，'
                  f'峰值显存 {gpu["memory_used_mib"]["maximum"]:,.0f} MiB，最高温度 '
                  f'{gpu["temperature_c"]["maximum"]:.0f} °C。']
    lines += ['', '![训练损失与梯度范数](training.png)', '',
              '每个损失点为对应更新内的动作损失均值，修复成功率由待运行的独立开发对照验收。',
              '', '[逐次更新](updates.csv) · [测量与硬件回执](measurements.json) · [训练登记](../../registration.json)']
    (output / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'updates':len(updates),'records':summary['records'],'pipeline_seconds':seconds,'audit_passed':True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(render(read(args.source), args.output), ensure_ascii=False))
