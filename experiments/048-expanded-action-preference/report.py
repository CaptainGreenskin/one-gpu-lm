"""Project an audited preference run into anonymous optimizer and hardware metrics."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
from statistics import fmean

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.runstate import identity_digest
from one_gpu_lm.training.preference_epoch import replay_history
from one_gpu_lm.verified.candidate_handoff import TRAINING_RECEIPTS
from one_gpu_lm.verified.recovery_report import hardware
from one_gpu_lm.verified.tasks import relative


CANDIDATE = '048-expanded-action-preference'
BEIJING = timezone(timedelta(hours=8))
AVERAGES = {'mean_loss': 'loss', 'mean_policy_margin': 'policy_margin',
            'mean_reference_margin': 'reference_margin'}
UPDATE_FIELDS = ('step', 'pairs', *AVERAGES, 'mean_margin_gain', 'grad_norm', 'seconds',
                 'peak_allocated_bytes', 'peak_reserved_bytes')
TRAINING_CHECKS = ('complete', 'all_selected_records_consumed', 'frozen_files_unchanged',
                   'frozen_parameters_unchanged', 'adapter_changed',
                   'saved_reload_greedy_ids_equal', 'saved_reload_action_scores_equal')


def read(path):
    return json.loads(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def update_metrics(selection, curve, updates):
    """Reconcile the entire scheduled pass before dropping per-pair fields."""
    batches = selection['batches']
    require(len(curve) == len(updates) == len(batches) == selection['updates']
            and sum(map(len, batches)) == selection['pairs']
            and selection['order_sha256'] == identity_digest(batches)
            and selection['final_test_opened'] is False, 'Optimizer windows must cover the exact registered pass')
    replay_history(batches, curve)
    rows = []
    for step, update in zip(curve, updates, strict=True):
        records = step['records']
        require(update == ({key: value for key, value in step.items() if key != 'records'}
                           | {'pairs': len(records)}) and records,
                'Aggregate optimizer receipt differs from its committed records')
        row = {key: update[key] for key in ('step', 'pairs', 'grad_norm', 'seconds',
                                            'peak_allocated_bytes', 'peak_reserved_bytes')}
        row.update({key: fmean(record[field] for record in records) for key, field in AVERAGES.items()})
        row['mean_margin_gain'] = fmean(r['policy_margin'] - r['reference_margin'] for r in records)
        require(0 <= row['peak_allocated_bytes'] <= row['peak_reserved_bytes'], 'Invalid GPU memory measurement')
        rows.append(row)
    return rows


def gpu_attempt(output, result):
    start, stop = (datetime.fromisoformat(result[key]).timestamp() for key in ('started_utc', 'finished_utc'))
    require(start <= stop, 'Invalid training attempt time interval')
    collector = result['telemetry']
    path = output / 'private' / f"attempt-{result['attempt']:04d}" / 'telemetry.csv'
    if not path.is_file():
        require(collector['samples'] == 0, 'Training attempt is missing its recorded GPU samples')
        return {'attempt': result['attempt'], 'samples': 0, 'pipeline_seconds': stop - start,
                'sampling_errors': collector['errors'], 'telemetry_available': False}
    with path.open(encoding='utf-8') as stream:
        sample_attempts = {int(row['attempt']) for row in csv.DictReader(stream)}
    require(sample_attempts <= {result['attempt']}, 'GPU samples belong to a different attempt')
    measured = hardware(path, {'started_epoch': start, 'finished_epoch': stop, 'telemetry': collector})
    require(measured['samples'] == collector['samples'], 'GPU sample count differs from the stopped collector')
    return {key: value for key, value in measured.items() if key != 'collector'} | {
        'attempt': result['attempt'], 'telemetry_available': True, 'telemetry_sha256': sha256(path),
        'sampling_errors': collector['errors'], 'collector_stopped': collector['thread_stopped']}


def attempt_metrics(job, output, result, protocol_hash):
    states, terminal = {}, {}
    attempts_dir = job / 'commands'
    for command in sorted(attempts_dir.glob('[0-9]*'), key=lambda path: int(path.name)):
        requested = read(command / 'request.json')
        action = requested['action']
        require(action in ('train', 'resume', 'audit') and requested['protocol_sha256'] == protocol_hash,
                'Unexpected command in the training history')
        evidence = command / 'state.json'
        state = read(evidence) if evidence.is_file() else {}
        if state:
            require(state['mode'] == action and state['protocol_sha256'] == protocol_hash
                    and state['final_test_opened'] is False, 'Training command differs from its registered run')
        error = state.get('error_type')
        require(error is None or re.fullmatch(r'[A-Z][A-Za-z0-9_]{0,59}', error),
                'Report only the exception class')
        counter = int(command.name)
        states[counter] = {'command': counter, 'action': action, 'finished': 'finished_utc' in state,
                           'exit_code': state.get('exit_code'), 'error_type': error}
        if 'attempt_result_sha256' in state:
            path = command / 'attempt-result.json'
            value = read(path)
            require(sha256(path) == state['attempt_result_sha256'] and value['protocol_sha256'] == protocol_hash
                    and value['final_test_opened'] is False and value['weights_public'] is False
                    and value['attempt'] not in terminal and action != 'audit',
                    'The retained attempt result differs from its command')
            terminal[value['attempt']] = value
            states[counter]['attempt'] = value['attempt']
            states[counter]['stage'] = value['stage']
    require(terminal.get(result['attempt'], result) == result, 'The final attempt differs from the training audit')
    terminal[result['attempt']] = result
    return list(states.values()), [gpu_attempt(output, value) for _, value in sorted(terminal.items())
                                  if 'finished_utc' in value and 'telemetry' in value]


def capture(root, protocol, publication):
    """Read private evidence on the workstation and return aggregates only."""
    plan, published = read(protocol), read(publication)
    identity = sha256(protocol)
    require(plan['candidate'] == CANDIDATE and published['protocol_sha256'] == identity
            and re.fullmatch(r'[0-9a-f]{40}', published['commit'])
            and published['anonymous_files_verified'] is True and published['ci_conclusion'] == 'success'
            and plan['final_test_opened'] is False and plan['weights_public'] is False,
            'Require the exact verified public training registration')
    output = root / relative(plan['run_directory'])
    integrity_path = output / 'integrity.json'
    integrity = read(integrity_path)
    audit = sha256(integrity_path)
    review = read(output / 'candidate-input-review.json')
    receipts = integrity['receipt_sha256']
    result = read(output / 'result.json')
    require(integrity['passed'] is True and integrity['protocol_sha256'] == identity
            and integrity['final_test_opened'] is False and integrity['weights_public'] is False
            and integrity['audit_optimizer_updates'] == integrity['audit_model_generations'] == 0
            and set(receipts) == TRAINING_RECEIPTS
            and all(sha256(output / name) == expected for name, expected in receipts.items())
            and review['training_integrity_sha256'] == audit and review['experiment'] == CANDIDATE
            and result['stage'] == 'completed' and result['protocol_sha256'] == identity
            and result['development_scores'] == 0 and result['final_test_opened'] is False
            and result['weights_public'] is False and all(result[key] is True for key in TRAINING_CHECKS),
            'Results must match the complete preference training audit')
    selection = read(output / 'selection.json')
    projected = update_metrics(selection, read(output / 'curve.json'), read(output / 'updates.json'))
    require(len(projected) == integrity['optimizer_updates'] == result['optimizer_steps']
            == plan['expected_optimizer_updates'] and sum(row['pairs'] for row in projected)
            == result['pairs_consumed'] == plan['planned_pairs']
            and selection['order_sha256'] == plan['order_sha256'] == result['order_sha256']
            and result['trainable_parameters'] == plan['expected_trainable_parameters'],
            'Complete training metrics must represent the exact registered work')
    commands, gpu = attempt_metrics(protocol.parent, output, result, identity)
    return {'schema_version': 1, 'protocol_sha256': identity, 'registration_commit': published['commit'],
            'integrity_sha256': audit, 'audit_passed': True, 'stage': 'completed',
            'started_utc': result['started_utc'], 'finished_utc': result['finished_utc'],
            'trainable_parameters': result['trainable_parameters'], 'order_sha256': plan['order_sha256'],
            'saved_reload_greedy_ids_equal': True, 'saved_reload_action_scores_equal': True,
            'attempt_count': result['attempt'], 'commands': commands, 'gpu_attempts': gpu, 'updates': projected,
            'final_test_opened': False, 'weights_public': False}


def figure(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5.5), layout='constrained')
    steps = [row['step'] for row in rows]
    for ax, key, label, color in zip(axes, ('mean_loss', 'mean_margin_gain'),
                                    ('Mean preference loss', 'Mean log-probability margin gain'),
                                    ('#185381', '#356C4C'), strict=True):
        ax.plot(steps, [row[key] for row in rows], marker='o', markersize=4, color=color)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.25)
    axes[1].axhline(0, color='#777777', linestyle=':', linewidth=1)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_xlabel('Optimizer update')
    fig.savefig(path, dpi=160, metadata={'Software': 'one-gpu-lm'})
    plt.close(fig)


def render(directory, registration):
    value = read(directory / 'measurements.json')
    plan = read(registration)
    rows = value['updates']
    require(value['protocol_sha256'] == plan['private_protocol_sha256'] and value['stage'] == 'completed'
            and value['audit_passed'] is True and value['final_test_opened'] is False
            and value['weights_public'] is False and value['order_sha256'] == plan['training']['order_sha256']
            and len(rows) == plan['training']['scheduled_updates'] > 0
            and sum(row['pairs'] for row in rows) == plan['training']['pairs']
            and all(set(row) == set(UPDATE_FIELDS) and type(row['step']) is int and row['step'] == index
                    and type(row['pairs']) is int and row['pairs'] > 0
                    and all(type(metric) in (int, float) and math.isfinite(metric) for metric in row.values())
                    for index, row in enumerate(rows, 1)), 'Public measurements differ from the registered training')
    with (directory / 'updates.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=UPDATE_FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    figure(directory / 'training.png', rows)
    gpu = value['gpu_attempts']
    sampled = [item for item in gpu if item['samples']]
    start, stop = (datetime.fromisoformat(value[key]).astimezone(BEIJING)
                   for key in ('started_utc', 'finished_utc'))
    text = ['# 048 · 首次动作偏好训练', '',
            f'LoRA 完成 **{len(rows)} 次已审计更新**，共使用 **{plan["training"]["pairs"]} 对**样本。'
            f'其中 045 提供 {plan["training"]["baseline_pairs"]} 对，047 提供 {plan["training"]["expansion_pairs"]} 对。',
            '', f'可训练参数为 {value["trainable_parameters"]:,} 个。'
            '重载适配器后的固定输入生成和动作分数均通过核对。', '',
            f'最后一次训练尝试为北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {stop:%Y-%m-%d %H:%M:%S}。'
            f'共记录 {value["attempt_count"]} 次训练尝试，{len(sampled)} 次具有可核对的硬件采样记录，'
            f'合计 {sum(item["samples"] for item in gpu)} 条采样，{sum(item["sampling_errors"] for item in gpu)} 次采样错误。']
    if sampled:
        peak = max(item['memory_used_mib']['maximum'] for item in sampled)
        temp = max(item['temperature_c']['maximum'] for item in sampled)
        text.append(f'采样峰值显存 {peak:,.0f} MiB，最高温度 {temp:.0f} °C。')
    text += ['', '曲线显示训练样本上的偏好损失，以及相对底座的偏好边际变化。模型效果由独立开发对照测量。',
             '', '![偏好训练曲线](training.png)', '',
             '[逐步统计](updates.csv) · [核验记录](measurements.json) · [训练登记](../../registration.json)']
    (directory / 'README.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    source = modes.add_parser('capture')
    for flag in ('root', 'protocol', 'publication', 'output'):
        source.add_argument('--' + flag, type=Path, required=True)
    target = modes.add_parser('render')
    target.add_argument('--results', type=Path, required=True)
    target.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'capture':
        write_json(args.output, capture(args.root, args.protocol, args.publication))
    else:
        render(args.results, args.registration)


if __name__ == '__main__':
    main()
