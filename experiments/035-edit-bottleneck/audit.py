"""Audit the base comparison and export patch-free, command-free diagnostics."""
import argparse
from collections import Counter
from pathlib import Path
import json

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.behavior_diagnosis import episode
from one_gpu_lm.verified.historic_outcomes import executed_completion
from one_gpu_lm.verified.historic_policy import feedback
from one_gpu_lm.verified.historic_replay import recorded_grade, replay
from one_gpu_lm.verified.historic_tasks import prepare
from one_gpu_lm.verified.resilient_eval_audit import audit_task
from one_gpu_lm.verified.rollout import normalization

from baseline import OUTPUT, PREVIOUS_RUN, inputs, read


def diagnostics(directory, trace, task, budget):
    tools = {step['call']: read(directory / f"tool-{step['call']:03d}.json")
             for step in trace['steps'] if 'observation' in step}
    value = episode(trace, tools, task['regression_command'], budget['max_tool_output_chars'])
    prefix = value['steps'][:value['calls_before_first_edit_or_all']]
    counts = Counter(step['kind'] for step in prefix)
    return {
        'before_first_edit_calls': len(prefix),
        'before_first_edit_actions': dict(sorted(counts.items())),
        'inspection_half_or_more': bool(prefix) and
            2 * counts['inspection_shaped_no_source_change'] >= len(prefix),
        'repeated_same_command_source_feedback': sum(s['repeated_command_source_feedback'] for s in prefix),
        'nonzero_exits': sum(s['tool_returncode'] not in (None, 0) for s in prefix),
        'compacted_calls': sum(s['context_dropped_rounds'] > 0 for s in prefix),
        'truncated_feedback': sum(s['tool_output_exceeds_feedback_cap'] for s in prefix),
        'generation_limit_calls': sum(s['hit_generation_limit'] for s in prefix),
    }


def audited_row(root, data, directory, result, task, source, protocol, profile, tokenizer, norm,
                journal, reconstruction, broken, backend):
    path = directory / 'trajectory.json'
    trace = read(path)
    if (sha256(path) != result['trajectory_sha256'] or read(directory / 'result.json') != result
            or not trace['complete'] or not trace['steps'] or trace['id'] != task['id']
            or trace['split'] != 'development' or trace['training_export_allowed']
            or trace['backend'].get('request_journal_version') != 1
            or not backend['ready_epoch'] <= trace['started_epoch'] <= trace['finished_epoch'] <= backend['stopped_epoch']):
        raise ValueError('Unexpected frozen trajectory identity')

    model_replay = replay(directory, trace, task, protocol, profile, tokenizer, norm)
    requests = audit_task(directory, trace, result, journal)
    grade = recorded_grade(root, data, directory, task, source, broken, reconstruction)
    submitted = executed_completion(trace, task, trace['final_source_sha256'])
    budget = protocol['evaluation_budget']
    first = trace['steps'][0]
    first_edit = next((s['call'] for s in trace['steps']
                      if 'source_after' in s and s['source_after'] != s['source_before']), None)
    within = (trace['actor_seconds'] <= budget['max_task_seconds']
              and trace['attempted_calls'] <= budget['max_model_calls']
              and trace['generated_tokens'] <= budget['max_total_generated_tokens'])
    success = grade['passed'] and within and submitted['submitted']
    if (not model_replay['passed'] or grade != trace['grade'] or grade != result['grade']
            or any(trace[k] != v or result[k] != v for k, v in submitted.items())
            or trace['actor_budget_compliant'] != within or result['actor_budget_compliant'] != within
            or result['complete_success_within_budget'] != success or result['first_edit_call'] != first_edit):
        raise ValueError('Replayed generation, grading, or budget differs')
    row = {
        'task_number': result['task_number'], 'arm': result['contrast_arm'], 'family': task['family'],
        'audit_passed': True, 'completed_repair': success,
        'correct_without_submission': grade['passed'] and within and not submitted['submitted'],
        'first_exact_regression': 'observation' in first and first.get('command') == task['regression_command'],
        'first_edit_call': first_edit, 'no_edit': first_edit is None,
        'attempted_calls': trace['attempted_calls'], 'input_tokens': model_replay['known_input_tokens'],
        'generated_tokens': trace['generated_tokens'], 'actor_seconds': trace['actor_seconds'],
        'timeout_recovered': result['timeout_call'] is not None,
        **diagnostics(directory, trace, task, budget),
    }
    return row, first['generation']['input_tokens_sha256'], requests


def totals(rows):
    groups = {}
    fields = (
        'completed_repair', 'correct_without_submission', 'first_exact_regression', 'no_edit',
        'attempted_calls', 'input_tokens', 'generated_tokens', 'timeout_recovered',
        'before_first_edit_calls', 'repeated_same_command_source_feedback', 'nonzero_exits',
        'compacted_calls', 'truncated_feedback', 'generation_limit_calls',
    )
    for arm in sorted({row['arm'] for row in rows}):
        selected = [row for row in rows if row['arm'] == arm]
        stuck = [row for row in selected if row['no_edit']]
        counts = Counter()
        for row in stuck:
            counts.update(row['before_first_edit_actions'])
        groups[arm] = {
            'tasks': len(selected),
            **{field: sum(row[field] for row in selected) for field in fields},
            'no_edit_action_counts': dict(sorted(counts.items())),
            'no_edit_inspection_half_or_more': sum(row['inspection_half_or_more'] for row in stuck),
            'no_edit_with_repeat': sum(row['repeated_same_command_source_feedback'] > 0 for row in stuck),
            'no_edit_with_compaction': sum(row['compacted_calls'] > 0 for row in stuck),
            'no_edit_at_call_limit': sum(row['attempted_calls'] == 12 for row in stuck),
        }
    return groups


def run(root):
    ctx = inputs(root)
    out = root / OUTPUT
    previous_runs = read(root / PREVIOUS_RUN / 'controller.json')['runs']
    prior = {(row['task_number'], row['arm']): row for row in ctx.legacy_proof['rows']}
    old_raw = {(r['task_number'], r['contrast_arm']): r for r in previous_runs}
    expected = {(r['task_number'], arm) for r in ctx.order for arm in ('control', 'masked')}
    if set(prior) != expected or set(old_raw) != expected or len(previous_runs) != len(expected):
        raise ValueError('Incomplete 034 comparison')
    controller_path = out / 'controller.json'
    controller = read(controller_path)
    if (not controller['complete'] or not controller['backend_stopped'] or not controller['frozen_inputs_unchanged']
            or controller['registration_sha256'] != sha256(ctx.plan_path)
            or controller['schedule_sha256'] != ctx.plan['schedule_sha256']
            or controller['final_test_opened'] or controller['optimizer_updates'] != 0
            or [r['run_id'] for r in controller['runs']] != [r['run_id'] for r in ctx.order]):
        raise ValueError('Complete registered baseline sweep required')
    backends = controller['backend_lifecycles']
    if len(backends) != 1 or backends[0]['role'] != 'base' or backends[0]['index'] != 1:
        raise ValueError('Expected one unadapted base load')
    backend = backends[0]
    runtime = out / 'private/backends/0001-base'
    health = read(runtime / 'served-artifact.json')
    if (health['adapter'] is not False or health.get('adapter_sha256', {}) or
            health['profile_sha256'] != ctx.protocol['evaluation_profile_sha256']
            or read(runtime.with_name('0001-base-environment.json')) != read(root / ctx.protocol['environment_path'])):
        raise ValueError('Unexpected loaded model or environment')
    journal = RequestJournal(runtime / 'requests')
    profile = read(root / ctx.protocol['evaluation_profile_path'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ctx.runtime['model'], local_files_only=True, trust_remote_code=False)
    norm = normalization(root)
    reconstruction = out / 'private/audit'
    reconstruction.mkdir()
    admission = {r['id']: r for r in read(ctx.data / 'admission.json')['rows']}
    rows = []
    seen = set()
    count = 0
    for item, result in zip(ctx.order, controller['runs']):
        number = item['task_number']
        task = ctx.tasks[item['task_id']]
        source = ctx.sources[task['source_id']]
        if result['task_number'] != number or result['contrast_arm'] != 'base' or result['backend_index'] != 1:
            raise ValueError('Wrong base task or backend index')
        directory = out / 'private/tasks' / item['run_id']
        snapshot = ctx.data / 'snapshots' / task['source_id']
        before = prepare(snapshot, reconstruction / 'broken' / item['run_id'], source, task)
        base, first_input, requests = audited_row(
            root, ctx.data, directory, result, task, source, ctx.protocol, profile, tokenizer, norm,
            journal, reconstruction / 'graded' / item['run_id'], before, backend)
        if requests & seen:
            raise ValueError('Duplicate durable generation receipt')
        seen.update(requests)
        base['multifile'] = admission[item['task_id']]['multifile']
        rows.append(base)
        for arm in ('control', 'masked'):
            raw = old_raw[number, arm]
            previous = prior[number, arm]
            path = root / PREVIOUS_RUN / 'private/tasks' / raw['run_id']
            if (not previous['audit_passed'] or previous['family'] != task['family']
                    or previous['multifile'] != base['multifile'] or previous['first_input_sha256'] != first_input
                    or sha256(path / 'trajectory.json') != raw['trajectory_sha256']):
                raise ValueError('Incomplete cross-run opening match or changed prior trace')
            trace = read(path / 'trajectory.json')
            if trace['id'] != task['id'] or trace['split'] != 'development' or trace['training_export_allowed']:
                raise ValueError('Mismatched prior trace')
            for step in trace['steps']:
                if 'observation' not in step:
                    continue
                tool = read(path / f"tool-{step['call']:03d}.json")
                _, receipt = feedback(tool['output'], norm, ctx.protocol['evaluation_budget']['max_tool_output_chars'])
                if receipt != step['normalization'] or {k:v for k,v in tool.items() if k != 'output'} != step['observation']:
                    raise ValueError('Prior raw tool output changed')
            rows.append({k:v for k,v in previous.items() if k != 'first_input_sha256'}
                        | diagnostics(path, trace, task, ctx.protocol['evaluation_budget']))
            count += previous['attempted_calls']
    if {p.name for p in journal.directory.iterdir() if p.is_dir()} != seen:
        raise ValueError('Unaccounted base generation')
    value = {
        'schema_version': 1, 'passed': True, 'registration_sha256': controller['registration_sha256'],
        'base_controller_sha256': sha256(controller_path), 'base_verified_generation_requests': len(seen),
        'prior_verified_generation_requests': count, 'matched_tasks': len(ctx.order),
        'first_inputs_matched': True, 'optimizer_updates': 0, 'final_test_opened': False,
        'started_epoch': controller['started_epoch'], 'finished_epoch': controller['finished_epoch'],
        'telemetry': controller['telemetry'], 'arms': totals(rows), 'rows': rows,
    }
    write_json(out / 'audit.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root.resolve())
    print(json.dumps({key:result[key] for key in ('passed', 'matched_tasks', 'base_verified_generation_requests')}))
