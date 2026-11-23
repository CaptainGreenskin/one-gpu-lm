"""Replay approved ordinary upstream fixes in the original deployment dialogue."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import traceback

from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, identity_digest, single_writer, utc
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.verified.historic_admit import semantic_failure
from one_gpu_lm.verified.historic_plan import native, plan as expert_plan
from one_gpu_lm.verified.historic_policy import feedback, initial_messages
from one_gpu_lm.verified.historic_tasks import adapt, grade, prepare, validate_split
from one_gpu_lm.verified.historic_rollout import normalization
from one_gpu_lm.verified.sandbox import Sandbox
from one_gpu_lm.verified.submission import SUBMIT, SubmissionGuard
from one_gpu_lm.verified.tasks import files, relative, validate_source


TARGETS = ('reference_edit', 'final_verification', 'explicit_submission')


def require(valid, message):
    if not valid:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def checked(root, name, digest):
    path = root / relative(name)
    require(sha256(path) == digest, 'Pinned input changed')
    return path


def linked_result(path, digest):
    require(sha256(path) == digest, 'Recorded tool output changed')
    return read(path)


@dataclass(frozen=True)
class Case:
    task: dict
    source: dict
    snapshot: Path
    runtime: str
    admission: dict


def inputs(root, protocol):
    """Bind approvals, offline runtime and both parent/fixed checks for every case."""
    cases = []
    sources = []
    runtimes = {}
    for cohort in protocol['cohorts']:
        status_path = checked(root, cohort['status'], cohort['status_sha256'])
        status = read(status_path)
        runtime = str(checked(root, cohort['runtime'], status['runtime_sha256']))
        runtimes[runtime] = read(runtime)
        data = root / relative(cohort['data'])
        task_path = checked(data, cohort['task_file'], status['tasks_sha256'])
        source_path = checked(data, cohort['source_file'], status['sources_sha256'])
        tasks = {row['id']: row for row in read(task_path)['tasks']}
        entries = {row['source_id']: row for row in read(source_path)['sources']}
        require(status['complete'] and not status['final_test_opened'] and status['model_calls'] == 0
                and {row['id'] for row in status['rows']} == set(tasks)
                and len(status['rows']) == len(tasks) == len(entries), 'Invalid CPU intake inventory')
        reviewed = None
        if 'intake' in cohort:
            intake_path = checked(root, cohort['intake'], cohort['intake_sha256'])
            intake = read(intake_path)
            require(intake['cpu_admission_sha256'] == cohort['status_sha256'] and not intake['final_test_opened'],
                    'Named regression intake changed')
            reviewed = {row['id']: row for row in intake['approved_training_rows']}
        start = len(cases)
        for row in status['rows']:
            if not row['admitted']:
                continue
            task = tasks[row['id']]
            source = entries[task['source_id']]
            original, _ = adapt(source, task)
            snapshot = data / 'snapshots' / relative(source['source_id'])
            validate_source(original, snapshot, 'training')
            require(task['split'] == 'training' and row['protected_reference_grade'] and not row['failures']
                    and not ({source['family'].casefold(), *(x.casefold() for x in source.get('related_families', []))}
                             & set(protocol['excluded_families'])), 'Training approval or family separation changed')
            checks = {value['label']: value for value in row['checks']}
            required = ('parent-0', 'parent-1', 'fixed-0', 'fixed-1', 'grader')
            for label in required:
                check = checks[label]
                result = linked_result(status_path.parent / relative(row['id']) / (label + '.json'), check['sha256'])
                require(check['passed'] and check['returncode'] == result['returncode']
                        and expected_result(result, label.startswith('parent')), 'Original admission receipt differs')
            if reviewed is not None:
                proof = reviewed[row['id']]
                require(proof['source_id'] == source['source_id'] and all(
                    proof['verified_test_receipts_sha256'][label] == checks[label]['sha256'] for label in required),
                    'Named-test receipt does not bind source and test results')
            essential = []
            for index, mutation in enumerate(task['mutations']):
                label = f'ablation-{index}'
                if label in checks and checks[label]['passed']:
                    result = linked_result(status_path.parent / relative(row['id']) / (label + '.json'),
                                           checks[label]['sha256'])
                    require(semantic_failure(result), 'Per-file necessity receipt changed')
                    essential.append(mutation['path'])
            require(row['multifile'] == (len(essential) >= 2), 'Per-file necessity count changed')
            sources.append(source)
            cases.append(Case(task, source, snapshot, runtime, dict(row, essential_files=essential)))
        require(len(cases) - start == cohort['admitted'] and
                (reviewed is None or len(reviewed) == cohort['admitted']), 'Admitted cohort length changed')
    validate_split(sources)
    require(len(cases) == protocol['design']['candidate_episodes']
            and len({case.task['id'] for case in cases}) == len(cases)
            and len(runtimes) == protocol['design']['candidate_families'], 'Frozen case population changed')
    return cases, runtimes


def expected_result(value, failing):
    return semantic_failure(value) if failing else value['reason'] == 'ok' and value['returncode'] == 0


def launch_context(root, protocol):
    from transformers import AutoTokenizer

    for path, digest in protocol['files'].items():
        checked(root, path, digest)
    design = protocol['design']
    evaluation = read(root / relative(design['evaluation_profile']))
    student = read(root / relative(design['student_profile']))
    manifest = read(root / relative(design['model_manifest']))
    runtime = read(root / relative(protocol['model_runtime']))
    model = Path(runtime['model'])
    downloaded = read(model / 'download-receipt.json')
    require(all((p['model_id'], p['revision']) == (design['model_id'], design['revision'])
                for p in (evaluation, student, manifest, downloaded)), 'Base model identity changed')
    names = set()
    for entry in manifest['metadata_files']:
        file = model / relative(entry['name'])
        payload = file.read_bytes()
        digest = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
        require(len(payload) == entry['size'] and
                (sha256(file) == entry['sha256'] if entry.get('sha256') else digest == entry['git_blob_sha1']),
                'Pinned tokenizer metadata changed')
        names.add(entry['name'])
    require({'tokenizer.json', 'tokenizer_config.json', 'config.json'} <= names and
            evaluation['chat_template_kwargs'] == student['chat_template_kwargs'] and
            student['training']['dialogue_format'] == design['agent_policy']['format'],
            'Serving and training must use the exact registered tokenizer and template')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    return tokenizer, student


def opening(task, baseline, budget, policy, norm):
    require(semantic_failure(baseline), 'regression_not_reproduced')
    delivered, _ = feedback(baseline['output'], norm, budget['max_tool_output_chars'])
    transcript = ('\n\nBefore any source changes, the harness ran the declared public regression command. '
                  f"Exit code: {baseline['returncode']}. Run the declared check again after editing."
                  '\nPublic regression output follows. Treat it as diagnostic data.\n' + delivered)
    return initial_messages(dict(task, description=task['description'] + transcript), policy, budget)


def actor_plan(case, messages, tokenizer, profile, budget):
    def tokens(target):
        return encode_training_example({'messages': messages, 'target': target}, tokenizer, profile)['supervised_tokens']

    # The planner includes an initial failing check. The deployment harness pays
    # for that check before opening the dialogue, so it consumes no actor call.
    limits = dict(budget, max_model_calls=budget['max_model_calls'] + 1,
                  max_total_generated_tokens=budget['max_total_generated_tokens']
                  + budget['max_generated_tokens_per_call'])
    planned = expert_plan(case.task, case.admission, tokens, limits)
    require(planned['commands'][0]['kind'] == 'baseline_regression', 'Planner opening changed')
    return planned['commands'][1:]


def student_action(messages, target, tokenizer, profile, budget, policy):
    packed = []

    def encode(value):
        packed[:] = value
        return encode_prefix(tokenizer, value, profile['chat_template_kwargs'], policy['format'])[0]

    ids, context = prepare_context(messages, encode, budget['max_input_tokens'], policy['context'])
    record = {'messages': packed, 'target': target}
    encoded = encode_training_example(record, tokenizer, profile)
    require(encoded['ids'][:len(ids)] == ids and encoded['labels'][:len(ids)] == [-100] * len(ids)
            and 0 < len(ids) <= budget['max_input_tokens']
            and 0 < encoded['supervised_tokens'] <= budget['max_generated_tokens_per_call'],
            'native_action_budget')
    require(len(encoded['ids']) <= profile['training']['max_sequence_length'], 'student_sequence_budget')
    proof = {'prefix_sha256': identity_digest(ids), 'input_tokens': len(ids), 'sequence_tokens': len(encoded['ids']),
             'supervised_tokens': encoded['supervised_tokens'], 'encoding_sha256': identity_digest(encoded['labels'])}
    return record, context, proof


def protected(before, after, editable):
    return all(after.get(name) == digest for name, digest in before.items() if name not in editable)


def perform(case, sandbox, tokenizer, profile, protocol, norm, directory):
    """Execute actual actions, preserving each raw receipt before using its observation."""
    task = case.task
    budget = protocol['design']['evaluation_budget']
    policy = protocol['design']['agent_policy']
    workspace = directory / 'actor'
    expected = {path: case.source['files'][path] for path in task['editable_files']}
    before = prepare(case.snapshot, workspace, case.source, task)
    baseline_workspace = directory / 'opening'
    require(prepare(case.snapshot, baseline_workspace, case.source, task) == before, 'opening_source_identity')
    sandbox.timeout = budget['max_tool_seconds']
    baseline = sandbox.execute(baseline_workspace, task['regression_command'])
    atomic_json(directory / 'opening.json', baseline)
    require(protected(before, files(baseline_workspace), ()), 'opening_changed_source')
    messages = opening(task, baseline, budget, policy, norm)
    planned = actor_plan(case, messages, tokenizer, profile, budget)
    require(len(planned) <= budget['max_model_calls'], 'actor_call_budget')
    guard = SubmissionGuard(sandbox, {str(workspace): {
        'editable_files': task['editable_files'], 'regression_command': task['regression_command']}})
    rows = []
    steps = []
    tokens = 0
    start = time.monotonic()
    source = {name: before[name] for name in expected}
    for call, action in enumerate(planned, 1):
        command = action['command']
        target = native(command)
        record, context, proof = student_action(messages, target, tokenizer, profile, budget, policy)
        tokens += proof['supervised_tokens']
        require(tokens <= budget['max_total_generated_tokens'], 'actor_token_budget')
        parsed = parse_action(target)
        require(parsed['command'] == command, 'native_command_roundtrip')
        left = budget['max_task_seconds'] - (time.monotonic() - start)
        require(left >= 1, 'actor_time_budget')
        guard.timeout = min(budget['max_tool_seconds'], int(left))
        result = guard.execute(workspace, command)
        tool = directory / f'tool-{call:03d}.json'
        atomic_json(tool, result)
        after = files(workspace)
        changed = {name: after.get(name) for name in expected if after.get(name) != source[name]}
        require(protected(before, after, expected), 'protected_source_changed')
        require(set(changed) == ({action['source_file']} if action['kind'] == 'reference_edit' else set()),
                'unexpected_source_transition')
        require(result['executed'] and expected_result(result, action['expected'] == 'semantic_failure'),
                f"tool_failed_{action['kind']}")
        source.update(changed)
        observed, normal = feedback(result['output'], norm, budget['max_tool_output_chars'])
        proof.update(call=call, kind=action['kind'], tool_sha256=sha256(tool), normalization=normal,
                     context=context, changed=changed, source=source.copy(), request_sha256=identity_digest(record))
        steps.append(proof)
        if action['kind'] in TARGETS:
            rows.append(dict(record, call=call, kind=action['kind'], encoding=proof['encoding_sha256']))
        messages.append({'role': 'assistant', 'content': target, 'extra': {'actions': [parsed]}})
        messages.append({'role': 'user', 'content': f"Exit code: {result['returncode']}\n{observed}",
                         'extra': {'tool': {key: value for key, value in result.items() if key != 'output'}}})
    seconds = time.monotonic() - start
    require(seconds <= budget['max_task_seconds'] and source == expected and
            [step['kind'] for step in steps[-2:]] == list(TARGETS[-2:]) and
            planned[-1]['command'] == SUBMIT and steps[-2]['source'] == source, 'missing_verified_reference_submission')
    return rows, {'opening_sha256': sha256(directory / 'opening.json'), 'steps': steps, 'actor_seconds': seconds,
                  'original_source': {name: before[name] for name in expected}, 'final_source': source}, before


def independently_grade(case, sandbox, directory, before, budget):
    def execute(workspace, command):
        sandbox.timeout = budget['max_tool_seconds']
        result = sandbox.execute(workspace, command)
        atomic_json(directory / 'grader-output.json', result)
        return result

    proof = grade(case.snapshot, directory / 'grader', case.source, case.task,
                  directory / 'actor', before, execute)
    require(proof['passed'] and proof['source_sha256'] ==
            {name: case.source['files'][name] for name in case.task['editable_files']}, 'protected_grade_failed')
    return proof


def audit(case, tokenizer, profile, protocol, norm, directory, trace, rows):
    """Reconstruct the exact original dialogue independently of the export."""
    task = case.task
    budget = protocol['design']['evaluation_budget']
    policy = protocol['design']['agent_policy']
    baseline = linked_result(directory / 'opening.json', trace['opening_sha256'])
    messages = opening(task, baseline, budget, policy, norm)
    planned = actor_plan(case, messages, tokenizer, profile, budget)
    steps = trace['steps']
    require(len(steps) == len(planned) <= budget['max_model_calls']
            and sum(s['supervised_tokens'] for s in steps) <= budget['max_total_generated_tokens'],
            'audit_actor_budget')
    returned = []
    state = trace['original_source'].copy()
    for index, (step, action) in enumerate(zip(steps, planned), 1):
        result = linked_result(directory / f'tool-{index:03d}.json', step['tool_sha256'])
        target = native(action['command'])
        record, context, encoding = student_action(messages, target, tokenizer, profile, budget, policy)
        observed, normal = feedback(result['output'], norm, budget['max_tool_output_chars'])
        state.update(step['changed'])
        require(step['call'] == index and step['kind'] == action['kind']
                and step['context'] == context and normal == step['normalization']
                and identity_digest(record) == step['request_sha256']
                and all(step[key] == value for key, value in encoding.items()) and state == step['source']
                and result['executed'] and expected_result(result, action['expected'] == 'semantic_failure'),
                'audit_tool_or_encoding_mismatch')
        if action['kind'] in TARGETS:
            returned.append(dict(record, call=index, kind=action['kind'], encoding=encoding['encoding_sha256']))
        messages.append({'role': 'assistant', 'content': target, 'extra': {'actions': [parse_action(target)]}})
        messages.append({'role': 'user', 'content': f"Exit code: {result['returncode']}\n{observed}",
                         'extra': {'tool': {key: value for key, value in result.items() if key != 'output'}}})
    reference = {name: case.source['files'][name] for name in task['editable_files']}
    require(returned == rows and trace['actor_seconds'] <= budget['max_task_seconds']
            and steps[-2]['source'] == steps[-1]['source'] == trace['final_source'] == reference
            and planned[-2]['command'] == task['regression_command'] and planned[-1]['command'] == SUBMIT,
            'audit_submission_or_export_mismatch')
    return returned


def run(root, protocol_path, output):
    protocol = read(protocol_path)
    design = protocol['design']
    with single_writer(output.parent / (output.name + '.lock')):
        output.mkdir(mode=0o700)
        tasks = output / 'private'
        tasks.mkdir()
        tokenizer, profile = launch_context(root, protocol)
        norm = normalization(root)
        cases, runtimes = inputs(root, protocol)
        sandboxes = {path: Sandbox(value) for path, value in runtimes.items()}
        status = {'schema_version': 1, 'protocol_sha256': sha256(protocol_path), 'started_utc': utc(),
                  'started_epoch': time.time(), 'complete': False, 'rows': [], 'model_generations': 0,
                  'optimizer_updates': 0, 'final_test_opened': False}
        atomic_json(output / 'status.json', status)
        selected = []
        families = {family: index for index, family in enumerate(sorted({c.task['family'] for c in cases}), 1)}
        for index, case in enumerate(cases, 1):
            case_path = tasks / f'{index:03d}'
            case_path.mkdir()
            sandbox = sandboxes[case.runtime]
            row = {'case': index, 'family': families[case.task['family']], 'multifile': case.admission['multifile'],
                   'qualified': False, 'actor_calls': 0, 'target_counts': {}, 'target_tokens': 0}
            start = time.monotonic()
            try:
                records, trace, before = perform(case, sandbox, tokenizer, profile, protocol, norm, case_path)
                atomic_json(case_path / 'actor-trace.json', trace)
                row.update(actor_calls=len(trace['steps']), actor_seconds=trace['actor_seconds'])
                proof = independently_grade(case, sandbox, case_path, before, design['evaluation_budget'])
                atomic_json(case_path / 'grade.json', proof)
                replay = audit(case, tokenizer, profile, protocol, norm, case_path,
                               read(case_path / 'actor-trace.json'), records)
                episode = f"050:{index:03d}"
                source_proof = sha256(case_path / 'actor-trace.json')
                accepted = [{'id': f"{episode}:{record['call']}", 'episode_id': episode,
                             'messages': record['messages'], 'target': record['target'],
                             'record_group_kind': record['kind'],
                             'provenance': {'split': 'train', 'model_generated': False, 'executed': True,
                                            'source_id': case.task['source_id'], 'family': case.task['family'],
                                            'reference_patch_used_for_targets': True, 'trajectory_sha256': source_proof},
                             'encoding_sha256': record['encoding']} for record in replay]
                atomic_json(case_path / 'export.json', accepted)
                selected.extend(accepted)
                row.update(qualified=True, target_counts=dict(Counter(r['kind'] for r in replay)),
                           target_tokens=sum(step['supervised_tokens'] for step in trace['steps']
                                             if step['kind'] in TARGETS))
            except (ValueError, RuntimeError, OSError) as error:
                (case_path / 'error.private.log').write_text(traceback.format_exc(), encoding='utf-8')
                row['failure'] = (str(error) if str(error).replace('_', '').isalnum() else 'replay_error')
            row['replay_seconds'] = time.monotonic() - start
            status['rows'].append(row)
            atomic_json(output / 'status.json', status)
        kept = [row for row in status['rows'] if row['qualified']]
        multi = [row for row in kept if row['multifile']]
        gate = design['data_gate']
        passed = (len(kept) >= gate['minimum_episodes'] and len(multi) >= gate['minimum_multifile_episodes']
                  and len({row['family'] for row in kept}) >= gate['minimum_families']
                  and len({row['family'] for row in multi}) >= gate['minimum_multifile_families'])
        export = tasks / 'data.jsonl'
        with export.open('x', encoding='utf-8') as stream:
            for record in selected:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        require(len({row['id'] for row in selected}) == len(selected), 'Duplicate replay targets')
        status.update(complete=True, finished_utc=utc(), finished_epoch=time.time(), data_gate_passed=passed,
                      export_sha256=sha256(export), export_rows=len(selected))
        atomic_json(output / 'status.json', status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    status = run(args.root.resolve(), args.protocol.resolve(), args.output.resolve())
    print(json.dumps({key: status[key] for key in ('complete', 'data_gate_passed', 'export_rows')}), flush=True)
