"""Render an audited grounded-edit training run from its anonymous measurements."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import json
from math import isfinite
from pathlib import Path
import runpy


EXPERIMENT = Path(__file__).resolve().parent
PARENT_REPORT = EXPERIMENT.parent/'042-first-edit-sft/report.py'
EVIDENCE = ('saved_reload_greedy_ids_equal', 'frozen_sample_unchanged', 'frozen_gradients_absent',
            'all_selected_records_consumed', 'frozen_files_unchanged')


def read(path):
    return json.loads(path.read_bytes())


def project(value, registration):
    metrics = value['metrics']
    planned = registration['selection']
    if (value['protocol_sha256'] != registration['protocol_sha256'] or not value['integrity_sha256']
            or value['final_test_opened'] or value['weights_public']
            or any(metrics[key] is not True for key in EVIDENCE)):
        raise ValueError('Require the completed audit and original registration for the training report')
    records = value['records']
    if ([row['record'] for row in records] != list(range(1,len(records)+1))
            or dict(Counter(row['record_group_kind'] for row in records)) != planned['target_counts']):
        raise ValueError('The executed order or target counts differ from the registered population')
    batches = defaultdict(list)
    for row in records:
        batches[row['update']].append(row)
    updates = []
    for number, raw in enumerate(value['updates'],1):
        batch = batches.pop(number, [])
        if not batch or raw['step'] != number or raw['records'] != len(batch):
            raise ValueError('An optimizer update differs from its executed record window')
        updates.append({'step':number,'records':len(batch),'input_tokens':sum(row['tokens'] for row in batch),
                        'supervised_tokens':sum(row['supervised_tokens'] for row in batch),
                        'mean_loss':sum(row['loss'] for row in batch)/len(batch),
                        'grad_norm':raw['grad_norm'],'seconds':raw['seconds']})
    totals = {key:sum(row[key] for row in updates) for key in ('records','input_tokens','supervised_tokens')}
    if (batches or len(updates) != planned['optimizer_updates'] or len(updates) != metrics['optimizer_steps']
            or any(totals[key] != planned[key] for key in totals)
            or metrics['action_records_consumed'] != totals['records']
            or any(metrics[key] != totals[key] for key in ('input_tokens','supervised_tokens'))
            or not all(isfinite(item) for row in updates for item in row.values())):
        raise ValueError('The anonymized updates do not reproduce the complete audited training run')
    return updates, totals


def render(directory, registration_path, *, title='052 · 源码证据微调结果'):
    value = read(directory/'measurements.json')
    registration = read(registration_path)
    updates, totals = project(value,registration)
    metrics, gpu = value['metrics'], value['hardware']
    zone = timezone(timedelta(hours=8))
    start, end = (datetime.fromtimestamp(metrics[key],zone) for key in ('started_epoch','finished_epoch'))
    seconds = (end-start).total_seconds()
    result = {'protocol_sha256':value['protocol_sha256'],'integrity_sha256':value['integrity_sha256'],
              **totals,'optimizer_updates':len(updates),'trainable_parameters':metrics['trainable_parameters'],
              'training_seconds':metrics['training_seconds'],'pipeline_seconds':seconds,
              'gpu_samples':gpu['samples'],'gpu_sampling_errors':gpu['collector']['errors'],
              'gpu_utilization_percent':gpu['utilization_percent'],'gpu_memory_used_mib':gpu['memory_used_mib'],
              'gpu_temperature_c':gpu['temperature_c'],'reload_output_matched':True,
              'final_test_opened':False,'weights_public':False}
    parent = runpy.run_path(str(PARENT_REPORT))
    with (directory/'updates.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,parent['UPDATE_FIELDS'],lineterminator='\n')
        writer.writeheader()
        writer.writerows(updates)
    parent['figure'](directory/'training.png',updates)
    (directory/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines = [f'# {title}', '',
             f'完成 **{len(updates)}** 次更新，共用 **{totals["records"]}** 个动作和 '
             f'**{totals["supervised_tokens"]:,}** 个监督 token。LoRA 可训练参数 {metrics["trainable_parameters"]:,}，'
             '保存后重载的贪心生成 ID 一致。训练产物通过独立审计。', '',
             f'北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}。'
             f'单卡流水线用时 {seconds:.0f} 秒，训练循环用时 {metrics["training_seconds"]:.0f} 秒。', '',
             f'GPU 采样 {gpu["samples"]} 条，失败 {gpu["collector"]["errors"]} 次。平均利用率 '
             f'{gpu["utilization_percent"]["mean"]:.1f}%，峰值显存 {gpu["memory_used_mib"]["maximum"]:,.0f} MiB，'
             f'最高温度 {gpu["temperature_c"]["maximum"]:.0f} °C。', '',
             '![训练损失与梯度范数](training.png)', '',
             '各点对应一次优化器更新。代码修复成功率由独立开发任务验收。', '',
             '[逐次更新](updates.csv) · [测量与硬件回执](measurements.json) · [训练登记](../../training-registration.json)']
    (directory/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--registration',type=Path,default=EXPERIMENT/'training-registration.json')
    default = parser.parse_args()
    print(json.dumps(render(default.directory,default.registration),allow_nan=False))
