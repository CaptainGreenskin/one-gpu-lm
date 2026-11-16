"""Read-only post-hoc action allocation diagnosis of an audited 025 run.

No model calls, command execution, training export or hidden-test access. Public
output contains numeric categories and hashes, never commands or model text.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shlex
import statistics

SUBMIT = 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inspection_shaped(command):
    """Conservative syntax label, not a proof of shell semantics or usefulness."""
    if any(s in command for s in ['`', '$(', '\n', '>', '<']):
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False
    segments = [[]]
    for token in tokens:
        if token in {';', '&&', '||', '|'}:
            segments.append([])
        else:
            segments[-1].append(token)
    has_read = False
    for segment in segments:
        if not segment:
            return False
        executable = segment[0]
        if executable in {'echo', 'cd'}:
            continue
        if executable == 'sed':
            # Only an explicit line-range print, with no in-place option.
            import re
            if len(segment) < 4 or segment[1] != '-n' or not re.fullmatch(r'[0-9]+(?:,[0-9]+)?p', segment[2]):
                return False
        elif executable not in {'cat', 'grep', 'rg', 'head', 'tail', 'wc', 'pwd', 'ls', 'nl'}:
            return False
        has_read = True
    return has_read


def episode(trace, tools, regression_command, feedback_cap):
    if not trace.get('complete') or trace.get('unknown_inflight_generation_cost'):
        raise ValueError('Only completed, accounted trajectories can be diagnosed')
    if len(trace['steps']) != trace['attempted_calls']:
        raise ValueError('Missing action steps')
    counters = Counter(); seen = set(); previous = None; calls = []
    pending_failed_check = False; successful_check = False
    ever_edited = False; pending_failed_check_after_edit = False
    for step in trace['steps']:
        call = step['call']; command = step.get('command')
        before = step['source_before']; after = step.get('source_after', before)
        changed = before != after
        tool = tools.get(call)
        rejected = step.get('observation', {}).get('executed') is False
        if not step['action_format_valid']:
            kind = 'invalid_format'
        elif rejected:
            kind = 'unexecuted_policy_rejection'
        elif tool is None:
            kind = 'no_tool_receipt'
        elif changed:
            kind = 'source_edit'
        elif command == SUBMIT:
            kind = 'explicit_submit'
        elif command == regression_command:
            kind = 'exact_regression'
        elif inspection_shaped(command):
            kind = 'inspection_shaped_no_source_change'
        else:
            kind = 'other_no_source_change'
        if changed and pending_failed_check:
            counters['edits_after_failed_exact_check'] += 1
            pending_failed_check = False
        if changed and pending_failed_check_after_edit:
            counters['corrections_after_failed_exact_check_of_edited_source'] += 1
            pending_failed_check_after_edit = False
        ever_edited |= changed
        if tool is not None and not rejected and command == regression_command:
            if tool['returncode'] != 0:
                pending_failed_check = True
                pending_failed_check_after_edit = ever_edited
                counters['failed_exact_regressions'] += 1
            else:
                successful_check = True
        generation = step['generation']
        key = None if command is None else (command, json.dumps(before, sort_keys=True),
                                            step.get('normalization', {}).get('delivered_sha256'))
        repeated = key is not None and key in seen
        if key is not None:
            seen.add(key)
        context = generation['context']
        item = {'call': call, 'kind': kind, 'source_changed': changed,
                'repeated_command_source_feedback': repeated,
                'consecutive_repeat': key is not None and key == previous,
                'input_tokens': generation['input_tokens'],
                'generated_tokens': generation['generated_tokens'],
                'hit_generation_limit': bool(generation['hit_generation_limit']),
                'context_dropped_rounds': context['dropped_rounds'],
                'tool_returncode': None if tool is None else tool['returncode'],
                'tool_output_exceeds_feedback_cap': tool is not None and len(tool['output']) > feedback_cap}
        calls.append(item); previous = key; counters[kind] += 1
    first_edit = next((c['call'] for c in calls if c['source_changed']), None)
    return {'calls': len(calls), 'first_edit_call': first_edit,
            'source_edit_calls': sum(c['source_changed'] for c in calls),
            'calls_before_first_edit_or_all': len(calls) if first_edit is None else first_edit - 1,
            'context_compacted_calls': sum(c['context_dropped_rounds'] > 0 for c in calls),
            'generation_limit_calls': sum(c['hit_generation_limit'] for c in calls),
            'feedback_truncated_calls': sum(c['tool_output_exceeds_feedback_cap'] for c in calls),
            'repeated_command_source_feedback_calls': sum(c['repeated_command_source_feedback'] for c in calls),
            'nonzero_tool_exits': sum(c['tool_returncode'] not in (None, 0) for c in calls),
            'successful_exact_check_observed': successful_check,
            'counts': dict(sorted(counters.items())), 'steps': calls}


def verify_tool_receipts(trace, tools, normalizer, cap):
    for step in trace['steps']:
        tool = tools.get(step['call'])
        if 'observation' not in step:
            if tool is not None:
                raise ValueError('Unexpected tool receipt')
            continue
        if tool is None or {k: v for k, v in tool.items() if k != 'output'} != step['observation']:
            raise ValueError('Tool metadata changed')
        raw = tool['output']
        prefix = raw if len(raw) <= cap else raw[:cap] + normalizer.MARKER
        _, receipt = normalizer.transform(prefix, 'extended', cap)
        if receipt != step['normalization']:
            raise ValueError('Tool text changed since the audited trajectory')


def training_allocation(audit, curve):
    """Align actual training losses with public action provenance and token counts."""
    records = []; episodes = []
    for row in audit['rows']:
        if not row['eligible_for_sft']:
            continue
        edits = [s['call'] for s in row['encoded_actions'] if s['kind'] == 'edit']
        episodes.append({'id': row['id'], 'repository': row['repository'],
                         'calls': row['attempted_calls'], 'first_edit': min(edits)})
        for step in row['encoded_actions']:
            records.append({'id': 'teacher-collection-001:' + row['id'] + ':' + str(step['call']), **step})
    records.sort(key=lambda r: digest(('025-student-v1\0' + r['id']).encode()))
    if len(records) != len(curve):
        raise ValueError('Training curve and action provenance differ')
    groups = defaultdict(list)
    for index, (record, point) in enumerate(zip(records, curve), 1):
        if point['record'] != index or any(record[k] != point[k] for k in ['tokens', 'supervised_tokens']):
            raise ValueError('Training order or encoding changed')
        groups[record['kind']].append(point)
    return {'full_episodes': len(episodes), 'action_records': len(records),
            'by_kind': {k: {'records': len(v), 'supervised_tokens': sum(r['supervised_tokens'] for r in v),
                           'mean_record_loss': statistics.mean(r['loss'] for r in v)} for k, v in sorted(groups.items())},
            'episodes_over_12_calls': sum(e['calls'] > 12 for e in episodes),
            'first_edit_after_8': sum(e['first_edit'] > 8 for e in episodes),
            'first_edit_after_12': sum(e['first_edit'] > 12 for e in episodes),
            'median_first_edit': statistics.median(e['first_edit'] for e in episodes),
            'median_episode_calls': statistics.median(e['calls'] for e in episodes)}


def run(root, out):
    from .rollout import normalization
    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError('Preserve the previous diagnosis')
    exp = root / 'experiments/025-verified-distillation'
    protocol = read(exp / 'evaluation-protocol.json')
    normalizer = normalization(root)
    def frozen():
        return all(digest((root / n).read_bytes()) == expected for n, expected in protocol['frozen_files'].items())
    if not frozen():
        raise ValueError('025 frozen inputs changed')
    all_rows = []
    for phase, directory, data, audit_name, budget_name in [
        ('training', 'teacher-collection-001', 'training-pool-data', 'collection-protocol.json', 'teacher_budget'),
        ('development', 'factorial-001', 'development-data', 'evaluation-protocol.json', 'evaluation_budget')]:
        run_dir = root / 'runs' / directory; audit = read(run_dir / 'audit.json')
        controller = read(run_dir / 'controller.json')
        if not controller.get('complete') or not controller.get('backend_stopped'):
            raise ValueError('Wait for the original run to finish')
        tasks = {t['id']: t for t in read(root / 'private' / data / 'tasks.json')['tasks']}
        cap = read(exp / audit_name)[budget_name]['max_tool_output_chars']
        if audit['protocol_sha256'] != digest((exp / audit_name).read_bytes()):
            raise ValueError('Audit belongs to a different protocol')
        if phase == 'development' and not audit['passed']:
            raise ValueError('Original evaluation audit must pass')
        if phase == 'training' and not all(r['input_replay_passed'] for r in audit['rows']):
            raise ValueError('Original collection input replay must pass')
        for row in audit['rows']:
            identity = row['id'] if phase == 'training' else row['run_id']
            location = run_dir / 'private/tasks' / identity; path = location / 'trajectory.json'
            if digest(path.read_bytes()) != row['trajectory_sha256']:
                raise ValueError('Audited trajectory changed')
            trace = read(path)
            tools = {s['call']: read(location / ('tool-%03d.json' % s['call']))
                     for s in trace['steps'] if (location / ('tool-%03d.json' % s['call'])).is_file()}
            verify_tool_receipts(trace, tools, normalizer, cap)
            value = episode(trace, tools, tasks[row['id']]['regression_command'], cap)
            value.update(id=row['id'], repository=row['repository'], phase=phase,
                         arm=row.get('arm', 'teacher'), repetition=row.get('repetition'),
                         trajectory_sha256=row['trajectory_sha256'],
                         success=trace['grade']['passed'] if phase == 'training' else row['success_within_budget'],
                         eligible_for_training=row.get('eligible_for_sft', False))
            all_rows.append(value)
    if len([r for r in all_rows if r['phase'] == 'training']) != 164 or len(all_rows) != 884:
        raise ValueError('Expected all 164 collection and 720 development trajectories')
    collection_audit = read(root / 'runs/teacher-collection-001/audit.json')
    allocation = training_allocation(collection_audit, read(root / 'runs/candidate-001/curve.json'))
    if not frozen():
        raise ValueError('025 inputs changed during read-only diagnosis')
    value = {'schema_version': 1, 'scope': 'post_hoc_diagnostic_not_a_new_evaluation',
             'source_protocol_sha256': digest((exp / 'evaluation-protocol.json').read_bytes()),
             'diagnosis_code_sha256': digest(Path(__file__).read_bytes()),
             'original_frozen_inputs_unchanged': True, 'rows': all_rows,
             'training_allocation': allocation, 'new_model_calls': 0, 'optimizer_updates': 0,
             'commands_executed': 0, 'hidden_checks_executed_or_used_as_features': False, 'final_test_opened': False,
             'raw_commands_text_or_weights_exported': False,
             'limitations': ['Inspection shape is a conservative syntax label, not proof of usefulness or shell semantics.',
                            'Exact-regression recovery counts exclude compound test commands; they are a lower bound.',
                            'Post-hoc associations do not identify the causal mechanism of the training regression.',
                            'Repeated runs are not independent tasks; development trajectories must never become training data.']}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); result = run(args.root, args.out)
    print(json.dumps({'trajectories': len(result['rows']), 'new_model_calls': 0, 'optimizer_updates': 0}))
