"""Project audited block receipts to anonymous data and paired-call curves."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import runpy

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.verified.recovery_report import hardware


DIR = Path(__file__).resolve().parent
BEIJING = timezone(timedelta(hours=8))
ARMS = ('greedy', 'sampled')
HARDWARE_KEYS = ('utilization_percent', 'power_w', 'memory_used_mib', 'temperature_c')


def read(path):
    return json.loads(path.read_bytes())


def block_hardware(out, controller):
    telemetry = out / 'telemetry.csv'
    measured = hardware(telemetry, controller)
    recorded = controller['telemetry']
    if measured['samples'] != recorded['samples'] or not recorded['thread_stopped']:
        raise ValueError('GPU samples do not match the stopped collector')
    fields = ('samples', 'first_to_last_seconds', 'maximum_gap_seconds', 'pipeline_seconds',
              'first_sample_delay_seconds', 'last_sample_to_finish_seconds', 'sampled_gpu_energy_kwh')
    return {**{name: measured[name] for name in (*fields, *HARDWARE_KEYS)},
            'sampling_errors': recorded['errors'], 'telemetry_sha256': sha256(telemetry)}


def capture(root, job, paired_report):
    """Check the global terminal receipt and each corresponding physical block."""
    code = runpy.run_path(str(job / 'source/experiments/047-expanded-action-forks/run.py'))
    protocol = job / 'protocol.json'
    publication = job / 'publication.json'
    plan, design, blocks = code['verified_campaign'](protocol, publication)
    output = root / code['OUTPUT']
    receipt = output / 'audit.json'
    terminal = read(receipt)
    schedule = {Path(where).name for _, where, _ in blocks}
    if (terminal['phase'] != 'audited' or terminal['active_block'] is not None
            or terminal['protocol_sha256'] != sha256(protocol) or set(terminal['blocks']) != schedule
            or not terminal['audited_attempts'] == terminal['expected_attempts'] == 2 * design['expected_training_states']
            or terminal['optimizer_updates'] or terminal['final_test_opened'] or terminal['weights_public']):
        raise ValueError('Require the complete stopped training-only campaign')
    rows, summaries = [], []
    for name, where, ordinals in blocks:
        out = root / where
        block_name = out.name
        block = job / name
        verified = code['audited_block'](out, plan['blocks'][name], ordinals)
        if sha256(block) != plan['blocks'][name] or terminal['blocks'][block_name] != verified:
            raise ValueError('Block evidence differs from the signed schedule or campaign audit')
        audit, controller = read(out / 'audit.json'), read(out / 'controller.json')
        if not (controller['phase'] == 'completed' and controller['complete'] and controller['training_only']
                and controller['frozen_inputs_unchanged'] and not controller['final_test_opened']
                and not controller['optimizer_updates']
                and controller['started_epoch'] <= controller['backend_ready_epoch']
                <= controller['finished_epoch'] <= controller['deadline_epoch']):
            raise ValueError('Block did not finish within its immutable execution boundary')
        rows.extend(audit['pairs'])
        summaries.append({'block': block_name, 'first_pair': ordinals[0], 'last_pair': ordinals[-1],
                          'protocol_sha256': sha256(block), 'controller_sha256': audit['controller_sha256'],
                          'started_epoch': controller['started_epoch'], 'finished_epoch': controller['finished_epoch'],
                          **verified, 'hardware': block_hardware(out, controller)})
    aliases = {name: f'F{number:02d}' for number, name in enumerate(sorted({row['family'] for row in rows}), 1)}
    return {'schema_version': 1, 'protocol_sha256': sha256(protocol), 'audit_sha256': sha256(receipt),
            'registration_commit': read(publication)['commit'], 'campaign_started_utc': terminal['started_utc'],
            'campaign_finished_utc': terminal['finished_utc'], 'started_epoch': summaries[0]['started_epoch'],
            'finished_epoch': summaries[-1]['finished_epoch'], 'blocks': summaries,
            'journal_requests_verified': terminal['verified_requests'], 'qualified_pairs': terminal['qualifying_preference_pairs'],
            'pairs': paired_report['anonymous_pairs'](rows, aliases), 'training_only': True, 'optimizer_updates': 0,
            'final_test_opened': False, 'weights_public': False}


def summarize(value, registration, paired_report):
    expected = {'protocol_sha256': registration['private_protocol_sha256'], 'training_states': registration['training_states'],
                'families': registration['training_families']}
    result = paired_report['summarize'](value, expected)
    blocks = value['blocks']
    if [block['pairs'] for block in blocks] != registration['block_sizes']:
        raise ValueError('Anonymous report must cover each registered block')
    start = 1
    previous_finish = value['started_epoch']
    for number, block in enumerate(blocks, 1):
        last = start + block['pairs'] - 1
        rows = value['pairs'][start - 1:last]
        successes = {arm: sum(row['arms'][arm]['verified_and_submitted'] for row in rows) for arm in ARMS}
        requests = sum(row['arms'][arm]['attempted_calls'] for row in rows for arm in ARMS)
        qualified = sum(row['contrast'] == 'matched_executed_contrast' for row in rows)
        if (block['block'] != f'block-{number:02d}' or block['first_pair'] != start or block['last_pair'] != last
                or block['attempts'] != 2 * block['pairs'] or block['successful_repairs'] != successes
                or block['requests'] != requests or block['preference_pairs'] != qualified
                or not previous_finish <= block['started_epoch'] < block['finished_epoch']
                or not math.isclose(block['hardware']['pipeline_seconds'], block['finished_epoch'] - block['started_epoch'])):
            raise ValueError('A displayed block does not reconcile to its audited pairs or timing')
        start = last + 1
        previous_finish = block['finished_epoch']
    if (blocks[0]['started_epoch'] != value['started_epoch'] or previous_finish != value['finished_epoch']
            or sum(block['requests'] for block in blocks) != value['journal_requests_verified']
            or sum(block['preference_pairs'] for block in blocks) != value['qualified_pairs']):
        raise ValueError('Aggregate data does not reconcile to all physical blocks')
    gpu = [block['hardware'] for block in blocks]
    samples = sum(item['samples'] for item in gpu)
    metrics = {}
    for field in HARDWARE_KEYS:
        if samples:
            metrics[field] = {'mean': sum(item[field]['mean'] * item['samples'] for item in gpu if item['samples']) / samples,
                              'maximum': max(item[field]['maximum'] for item in gpu if item['samples'])}
    result.update(audited_blocks=len(blocks), hardware=dict(metrics, samples=samples,
                  sampling_errors=sum(item['sampling_errors'] for item in gpu),
                  maximum_gap_seconds=max(item['maximum_gap_seconds'] for item in gpu),
                  sampled_gpu_energy_kwh=sum(item['sampled_gpu_energy_kwh'] for item in gpu),
                  active_pipeline_seconds=sum(item['pipeline_seconds'] for item in gpu),
                  covered_sampling_seconds=sum(item['first_to_last_seconds'] for item in gpu)))
    return result


def render(directory, registration, paired_report):
    value = read(directory / 'measurements.json')
    summary = summarize(value, registration, paired_report)
    write_json(directory / 'summary.json', summary)
    paired_report['write_curves'](directory, value['pairs'], salt='047-expanded-action-forks')
    start, end = (datetime.fromtimestamp(value[key], BEIJING) for key in ('started_epoch', 'finished_epoch'))
    gpu, outcomes, arms = summary['hardware'], summary['outcomes'], summary['arms']
    text = [
        '# 047 · 扩大同一状态的首次动作配对', '',
        f'在 {summary["paired_states"]} 对训练状态上，贪心组修对并提交 **{arms["greedy"]["completed_repairs"]}** 次，'
        f'首步采样组 **{arms["sampled"]["completed_repairs"]}** 次。得到 **{summary["qualified_pairs"]}** 对可用偏好样本。', '',
        f'两组都成功 {outcomes.get("both", 0)} 对，都失败 {outcomes.get("neither", 0)} 对。'
        f'贪心组单独成功 {outcomes.get("greedy_only", 0)} 对，首步采样组单独成功 {outcomes.get("sampled_only", 0)} 对。', '',
        f'北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}。'
        f'{summary["audited_blocks"]} 批完成审计，共核对 {summary["rollouts"]} 次执行和 {summary["verified_requests"]} 条请求。', '',
        '| 指标 | 贪心 | 首步采样 |', '|---|---:|---:|',
    ]
    for label, field in (('全程未改码', 'never_edited'), ('新调用数', 'attempted_calls'), ('新生成 token', 'generated_tokens')):
        text.append(f'| {label} | {arms["greedy"][field]:,} | {arms["sampled"][field]:,} |')
    text += ['', f'各批 GPU 流程合计 {gpu["active_pipeline_seconds"] / 3600:.2f} 小时。'
             f'硬件采样 {gpu["samples"]:,} 条，失败 {gpu["sampling_errors"]:,} 次，'
             f'批内最长采样间隔 {gpu["maximum_gap_seconds"]:.2f} 秒。']
    if gpu['samples']:
        text.append(f'平均利用率 {gpu["utilization_percent"]["mean"]:.1f}%，'
                    f'平均功耗 {gpu["power_w"]["mean"]:.0f} W，峰值显存 {gpu["memory_used_mib"]["maximum"]:.0f} MiB，'
                    f'最高温度 {gpu["temperature_c"]["maximum"]:.0f} °C。')
    text += ['', '本轮比较训练状态上两种采集方式。候选模型的质量由后续同预算的独立开发对照验证。', '',
             '![首次改码和修对提交曲线](call-curves.svg)', '',
             '[匿名逐次数据](matched-actions.csv) · [分批及硬件记录](measurements.json) · [配对统计](summary.json) · '
             '[预登记](../../registration.json)']
    (directory / 'README.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    helpers = runpy.run_path(str(DIR.parent / '045-first-action-forks/report.py'))
    print(json.dumps(render(args.results, read(DIR / 'registration.json'), helpers), allow_nan=False))
