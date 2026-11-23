"""Replay verified training repairs with the first edit's source visible to the model."""

import argparse
import ast
from collections import Counter
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
import shlex
import sys
import time
import traceback

from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, single_writer, utc
from one_gpu_lm.verified.tasks import relative


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def read(path):
    return json.loads(path.read_bytes())


def edit_arguments(shell):
    """Read literal arguments from the already reviewed expert edit, without executing it."""
    lines = shell.splitlines()
    opening = shlex.split(lines[0])
    require(len(opening) == 3 and opening[:2] == ['python3', '-']
            and opening[2] == ('<<' + lines[-1]), 'unsupported_reference_edit')
    tree = ast.parse('\n'.join(lines[1:-1]))
    arguments = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id in {'p', 'changes', 'edits'}:
            value = statement.value.args[0] if target.id == 'p' else statement.value
            arguments[target.id] = ast.literal_eval(value)
    require(set(arguments) in ({'p', 'changes'}, {'p', 'edits'}), 'unsupported_reference_edit')
    return arguments


def inspection(task, edit):
    arguments = edit_arguments(edit['command'])
    path = arguments['p']
    require(path == edit['source_file'] and str(relative(path)) in task['editable_files'],
            'invalid_first_edit_source')
    changes = {item['path']: item['after'] for item in task['mutations']}
    source = changes[path]
    lines = source.splitlines(keepends=True)
    visible = []
    numbers = set()
    if 'changes' in arguments:
        for before, _ in arguments['changes']:
            require(before and source.count(before) == 1, 'first_edit_anchor_not_in_original_source')
            start = source.index(before)
            first = source.count('\n', 0, start) + 1
            last = first + before.rstrip('\r\n').count('\n')
            numbers.update(range(first, last + 1))
            visible.append(before)
    else:
        for start, stop, _ in arguments['edits']:
            require(0 <= start <= stop <= len(lines), 'invalid_first_edit_range')
            # Include a neighboring line on inserts to identify the edit location.
            first, last = (start + 1, stop) if start < stop else (max(1, start), min(len(lines), start + 1))
            require(first <= last, 'first_edit_has_no_source_evidence')
            numbers.update(range(first, last + 1))
        visible.extend(f'{number}\n{lines[number - 1]}' for number in sorted(numbers))
    require(visible and numbers, 'first_edit_has_no_source_evidence')
    numbered = 'edits' in arguments
    ranges = []
    for number in sorted(numbers):
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    formatter = '{=;p;}' if numbered else 'p'
    program = ';'.join(f'{first},{last}{formatter}' for first, last in ranges)
    command = f'sed -n {shlex.quote(program)} {shlex.quote(path)}'
    return command, visible


def selected_cases(root, plan, prior):
    seen = set()
    selected = []
    for block in plan['blocks']:
        protocol = prior.checked(root, block['protocol'], block['protocol_sha256'])
        original = read(protocol)
        for path, checksum in original['files'].items():
            prior.checked(root, path, checksum)
        prior.checked(root, block['export'], block['export_sha256'])
        status = read(prior.checked(root, block['status'], block['status_sha256']))
        require(status['complete'] and status['protocol_sha256'] == block['protocol_sha256']
                and status['export_sha256'] == block['export_sha256']
                and not status['final_test_opened'], 'unverified_original_closure_replay')
        cases, runtimes = prior.inputs(root, original)
        matched = {case.task['source_id']: case for case in cases}
        require(not (seen & block['selection'].keys()) and block['selection'].keys() <= matched.keys(),
                'training_sources_overlap_or_are_missing')
        for source_id, choice in block['selection'].items():
            case = matched[source_id]
            selected.append((case, runtimes[case.runtime], original, choice))
        seen.update(block['selection'])
    require(len(selected) == plan['design']['candidate_episodes'], 'training_source_inventory_changed')
    return selected


def replay_one(case, sandbox, tokenizer, profile, protocol, norm, folder, planner, choice, prior):
    def visible_first_edit(case, messages, tokenizer, profile, budget):
        plan = planner(case, messages, tokenizer, profile, budget)
        require(plan[0]['kind'] == 'focused_inspection' and plan[1]['kind'] == 'reference_edit',
                'missing_first_inspection_or_edit')
        require(hash_bytes(plan[1]['command'].encode()).hexdigest() == choice['first_edit_sha256'],
                'first_edit_differs_from_the_accepted_demonstration')
        command, evidence = inspection(case.task, plan[1])
        require(sum(len(piece) for piece in evidence) <= budget['max_tool_output_chars'],
                'first_edit_evidence_exceeds_output_budget')
        plan[0] = dict(plan[0], command=command)
        return plan

    records, trace, before = prior.perform(case, sandbox, tokenizer, profile, protocol, norm,
                                           folder, visible_first_edit)
    atomic_json(folder/'actor-trace.json', trace)
    first = records[0]
    command, evidence = inspection(case.task, {'command': parse_action(first['target'])['command'],
                                               'source_file': next(iter(trace['steps'][1]['changed']))})
    require(first['kind'] == 'reference_edit' and first['call'] == 2
            and first['messages'][2]['extra']['actions'] == [{'command': command}]
            and all(piece in first['messages'][3]['content'] for piece in evidence),
            'exact_first_edit_source_not_visible_in_model_feedback')
    grade = prior.independently_grade(case, sandbox, folder, before, protocol['design']['evaluation_budget'])
    atomic_json(folder/'grade.json', grade)
    replayed = prior.audit(case, tokenizer, profile, protocol, norm, folder, trace, records, visible_first_edit)
    return replayed, trace, len(evidence)


def run(root, protocol_path, out):
    plan = read(protocol_path)
    source = root/relative(plan['closure_replay_code'])
    sys.path.insert(0, str(source))
    import run as prior
    from supplement import make_plan as bounded_lines
    from one_gpu_lm.verified.sandbox import Sandbox

    prior.checked(root, str((source/'run.py').relative_to(root)), plan['runner_sha256'])
    prior.checked(root, str((source/'supplement.py').relative_to(root)), plan['supplement_sha256'])
    with single_writer(out.parent/(out.name+'.lock')):
        out.mkdir(mode=0o700)
        private = out/'private'
        private.mkdir(mode=0o700)
        cases = selected_cases(root, plan, prior)
        signature = cases[0][2]['design']
        require(all(all(original['design'][key] == signature[key] for key in (
            'evaluation_budget', 'model_id', 'revision', 'evaluation_profile', 'student_profile', 'agent_policy'))
            for _, _, original, _ in cases), 'mixed_runtime_or_token_budget')
        tokenizer, profile = prior.launch_context(root, cases[0][2])
        norm = prior.normalization(root)
        families = {value: idx for idx, value in enumerate(sorted({case.task['family'] for case, *_ in cases}), 1)}
        state = {'schema_version': 1, 'protocol_sha256': sha256(protocol_path), 'started_utc': utc(),
                 'rows': [], 'complete': False, 'model_generations': 0, 'optimizer_updates': 0,
                 'final_test_opened': False, 'weights_public': False}
        atomic_json(out/'status.json', state)
        sandboxes, selected = {}, []
        for index, (case, runtime, original, choice) in enumerate(cases, 1):
            folder = private/f'{index:03d}'
            folder.mkdir()
            if case.runtime not in sandboxes:
                sandboxes[case.runtime] = Sandbox(runtime)
            row = {'case': index, 'family': families[case.task['family']], 'multifile': case.admission['multifile'],
                   'qualified': False, 'target_counts': {}, 'target_tokens': 0}
            start = time.monotonic()
            try:
                planner = bounded_lines if choice['line_ranges'] else prior.actor_plan
                records, trace, evidence_count = replay_one(case, sandboxes[case.runtime], tokenizer, profile,
                                                            original, norm, folder, planner, choice, prior)
                episode = f'052:{index:03d}'
                provenance = {'split': 'train', 'model_generated': False, 'executed': True,
                              'source_id': case.task['source_id'], 'family': case.task['family'],
                              'reference_patch_used_for_targets': True,
                              'trajectory_sha256': sha256(folder/'actor-trace.json')}
                exported = [{'id': f"{episode}:{record['call']}", 'episode_id': episode,
                             'messages': record['messages'], 'target': record['target'],
                             'record_group_kind': record['kind'], 'provenance': provenance,
                             'encoding_sha256': record['encoding']} for record in records]
                atomic_json(folder/'export.json', exported)
                selected.extend(exported)
                kinds = Counter(record['kind'] for record in records)
                row.update(qualified=True, actor_calls=len(trace['steps']), source_evidence_fragments=evidence_count,
                           target_counts=dict(kinds), target_tokens=sum(
                               step['supervised_tokens'] for step in trace['steps'] if step['kind'] in kinds))
            except (ValueError, RuntimeError, OSError) as error:
                (folder/'error.private.log').write_text(traceback.format_exc(), encoding='utf-8')
                row['failure'] = str(error) if str(error).replace('_', '').isalnum() else 'replay_error'
            row['replay_seconds'] = time.monotonic() - start
            state['rows'].append(row)
            atomic_json(out/'status.json', state)
        kept = [row for row in state['rows'] if row['qualified']]
        multifile = [row for row in kept if row['multifile']]
        gate = plan['design']['data_gate']
        passed = (len(kept) >= gate['minimum_episodes'] and len(multifile) >= gate['minimum_multifile_episodes']
                  and len({row['family'] for row in kept}) >= gate['minimum_families']
                  and len({row['family'] for row in multifile}) >= gate['minimum_multifile_families'])
        export = private/'data.jsonl'
        with export.open('x', encoding='utf-8') as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        state.update(complete=True, finished_utc=utc(), data_gate_passed=passed,
                     export_sha256=sha256(export), export_rows=len(selected))
        atomic_json(out/'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'protocol', 'out'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    value = run(args.root.resolve(), args.protocol.resolve(), args.out.resolve())
    print(json.dumps({key: value[key] for key in ('complete','data_gate_passed','export_rows')}))
