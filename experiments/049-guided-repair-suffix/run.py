"""Collect paired training-only continuations at the audited six-call boundary."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
from shutil import copyfile
from tempfile import TemporaryDirectory
import time

from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, gpu_lock_path, identity_digest, single_writer, utc
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.continuation_training_data import BASELINE, BASELINE_REVIEW, SECRET_SHAPES
from one_gpu_lm.verified.correction_rollout import advance, finish
from one_gpu_lm.verified.decision_state_audit import check_arm, packed_history
from one_gpu_lm.verified.decision_state_collection import DecisionCase, initialize, read, require
from one_gpu_lm.verified.first_edit_continuation import message_hash
from one_gpu_lm.verified.guard_audit import target_is_executed
from one_gpu_lm.verified.recovery_report import hardware
from one_gpu_lm.verified.recovery_sft_projection import ORIGINAL_PROTOCOL, ReplayRuntime
from one_gpu_lm.verified.recovery_state import copy_tree, inventory
from one_gpu_lm.verified.resilient_backend_control import JournalBackend
from one_gpu_lm.verified.rollout import normalization
from one_gpu_lm.verified.sandbox import Sandbox
from one_gpu_lm.verified.submission import SUBMIT
from one_gpu_lm.verified.tasks import files, relative, validate_source


DIR = Path(__file__).resolve().parent
DESIGN = DIR / 'design.json'
INTAKE = Path('private/049-no-edit-prefix-intake-v1.json')
REPLAY = Path('private/049-no-edit-tokenizer-replay-v1.json')
INITIAL = Path('private/first-action-045-retry-002/protocol.json')
RUNTIME = Path('private/parent-runtime.json')
OUTPUT = Path('runs/q049-guided')
ARMS = ('control', 'guided')
CELL_FIELDS = ('attempted_calls', 'input_tokens', 'generated_tokens', 'first_edit_call',
               'independently_correct', 'verified_and_submitted', 'invalid_actions',
               'timeout_recovered', 'actor_seconds')


def schedule(size):
    for number in range(1, size + 1):
        order = ARMS if number % 2 else ARMS[::-1]
        for arm in order:
            yield number, arm, f'{number:02d}-{arm}'


def with_guide(messages, guide):
    require(messages[0]['role'] == 'system' and guide not in messages[0]['content'],
            'Add the frozen guide to the original system message once')
    guided = deepcopy(messages)
    guided[0]['content'] += '\n\n' + guide
    return guided


def tokenize(messages, profile, policy, budget, tokenizer):
    def encode(history):
        return encode_prefix(tokenizer, history, profile['chat_template_kwargs'], policy['format'])[0]

    ids, context = prepare_context(messages, encode, budget['max_input_tokens'], policy['context'])
    require(context['dropped_rounds'] == 0 and 0 < len(ids) <= budget['max_input_tokens'],
            'Both registered openings must retain the entire recorded history')
    return {'prompt_tokens': len(ids), 'prompt_ids_sha256': identity_digest(ids),
            'packed_messages_sha256': message_hash(messages)}


def load_tokenizer(root):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(read(root / RUNTIME)['model'], local_files_only=True,
                                         trust_remote_code=False)


def reviewed_corpus(root, initial):
    historical = read(root / ORIGINAL_PROTOCOL)
    data = Path(initial['collection']) / relative(historical['data_directory'])
    review = read(root / BASELINE_REVIEW)
    require(sha256(data / 'tasks.json') == review['task_manifest_sha256']
            and sha256(data / 'sources.json') == review['source_manifest_sha256'],
            'The exact approved training corpus is required')
    tasks = {item['id']: item for item in read(data / 'tasks.json')['tasks']}
    sources = {item['repository']: item for item in read(data / 'sources.json')['sources']}
    return data, tasks, sources


def prepare_case(root, stage, row, number, proof, task, source, profile, tokenizer, design):
    trace_path, request_path = (root / relative(row[name]) for name in ('trace_path', 'request_path'))
    actor = trace_path.parent
    trace, request = read(trace_path), read(request_path)
    call = int(request_path.stem.split('-')[-1])
    step = trace['steps'][call - 1]
    state_path = actor / 'source-states' / (row['state_sha256'] + '.json')
    physical = read(state_path)
    before = read(actor / 'before.json')
    require(task['id'] == trace['id'] == row['task_id']
            and task['split'] == trace['split'] == source['split'] == 'training'
            and task['family'] == source['family'] == row['training_family']
            and source['revision'] == task['revision'] == row['source_commit']
            and trace['complete'] and not trace['eligible_verified_trajectory']
            and sha256(trace_path) == row['trace_sha256']
            and request_path.parent == actor and sha256(request_path) == step['request_sha256'] == row['request_sha256']
            and sha256(state_path) == step['source_evidence_before'] == row['state_sha256']
            and step['protected_before'] is True
            and physical['workspace_inventory'] == before == files(actor / 'workspace')
            and physical['source_sha256'] == trace['original_source_sha256'] == step['source_before']
            and set(step['source_before']) == set(task['editable_files'])
            and not set(task['hidden_checks']).intersection(before), 'The untouched training boundary changed')

    first_messages = request['messages']
    # The existing budget was charged once for each of the six archived requests.
    budget = dict(design['base_budget'], **row['remaining'])
    require(row['charged_calls'] == design['consumed_calls']
            and budget['max_model_calls'] == design['remaining_calls']
            and budget['max_total_generated_tokens'] ==
                design['base_budget']['max_total_generated_tokens'] - row['charged_generated_tokens']
            and request['max_input_tokens'] == budget['max_input_tokens']
            and request['max_new_tokens'] == min(budget['max_total_generated_tokens'],
                                                budget['max_generated_tokens_per_call'])
            and len(first_messages) == 2 + 2 * row['charged_calls'], 'The archived budget or history changed')
    policy = request['agent_policy']
    require(policy == design['base_policy'], 'Use the frozen native tool policy')
    messages = {'control': first_messages, 'guided': with_guide(first_messages, design['guide'])}
    openings = {arm: tokenize(value, profile, policy, budget, tokenizer) for arm, value in messages.items()}
    initial = openings['control']
    require(initial['prompt_tokens'] == proof['input_tokens'] == step['generation']['input_tokens']
            and initial['prompt_ids_sha256'] == proof['input_tokens_sha256'] == step['generation']['input_tokens_sha256']
            and initial['packed_messages_sha256'] == proof['packed_messages_sha256'],
            'The control must reproduce the original next input exactly')

    snapshot = stage / 'snapshots' / f'{number:02d}'
    snapshot.mkdir(parents=True)
    manifest = inventory(actor / 'workspace')
    require({name: item['sha256'] for name, item in manifest.items() if item['kind'] == 'file'} == before,
            'The copied bytes must match the six-call workspace inventory')
    copy_tree(actor / 'workspace', snapshot / 'workspace', manifest)
    atomic_json(snapshot / 'manifest.json', manifest)
    copyfile(actor / 'before.json', snapshot / 'before.json')
    atomic_json(snapshot / 'prefix.json', {'kind': 'early_edit', 'verified_source': None,
                'original_source_sha256': step['source_before'], 'origin_request_sha256': row['request_sha256'],
                'origin_state_sha256': row['state_sha256'], 'charged_calls': row['charged_calls'],
                'charged_generated_tokens': row['charged_generated_tokens'], 'messages': messages,
                'openings': openings, 'budget': budget})
    return {'number': number, 'task_id': task['id'], 'family': task['family'],
            'snapshot': f'snapshots/{number:02d}', 'added_input_tokens':
                openings['guided']['prompt_tokens'] - initial['prompt_tokens']}


def freeze(args):
    root, plan = args.root, args.plan
    design, initial, intake, replay = (read(path) for path in (DESIGN, root / INITIAL, root / INTAKE, root / REPLAY))
    source_contract = root / 'experiments/044-deployment-recovery-sft/design.json'
    parent = read(source_contract)
    require(sha256(root / INTAKE) == design['intake_sha256'] == replay['intake_sha256']
            and sha256(root / REPLAY) == design['tokenizer_replay_sha256']
            and intake['only_reviewed_training_sources'] is True
            and intake['development_trajectories_used_for_training'] is replay['development_used_for_training'] is False
            and intake['final_test_opened'] is replay['final_test_opened'] is design['final_test_opened'] is False
            and intake['new_model_generations'] == replay['new_model_generations'] == 0,
            'Require the registered physical intake and CPU token replay')
    rows = intake['prefixes']
    proofs = {row['request_sha256']: row for row in replay['proofs']}
    require(len(rows) == len(proofs) == replay['verified_prefixes'] == design['training_states']
            and len({row['task_id'] for row in rows}) == len(rows)
            and len({row['training_family'] for row in rows}) == design['training_families'],
            'Use every disjoint admitted prefix from the registered cohort')
    data, tasks, sources = reviewed_corpus(root, initial)
    expected = {name: Path(path) for name, path in initial['deployment_inputs'].items()}
    expected['sandbox'] = Path(initial['collection']) / 'private/task-runtime.json'
    student_path = root / relative(read(root / BASELINE)['profile_path'])
    profile = read(expected['evaluation_profile'])
    require(sha256(expected['evaluation_profile']) == replay['profile_sha256'] == design['profile_sha256']
            and (profile['model_id'], profile['revision']) == (design['model_id'], design['base_revision'])
            and not profile.get('generation_modes')
            and profile['chat_template_kwargs'] == read(student_path)['chat_template_kwargs']
            and parent['actor_budget']['max_generated_tokens_per_call'] == design['max_generated_tokens_per_call'],
            'Retain the pinned greedy base and original actor ceilings')
    for path in expected.values():
        require(sha256(path) == initial['runtime_inputs'][str(path)], 'Deployment or task runtime changed')
    visited = set()
    tokenizer = load_tokenizer(root)
    case_design = dict(design, base_budget=parent['actor_budget'], base_policy=parent['agent_policy'])
    with single_writer(plan.parent.with_suffix('.lock')):
        require(not plan.parent.exists() and not (root / OUTPUT).exists(), 'Preserve the first registration')
        with TemporaryDirectory(prefix='.guided-prepare-', dir=plan.parent.parent) as folder:
            stage = Path(folder) / 'prepared'
            stage.mkdir()
            for name, path in expected.items():
                copyfile(path, stage / (name + '.json'))
            selected = []
            for number, row in enumerate(rows, 1):
                task = tasks[row['task_id']]
                source = sources[task['repository']]
                if task['repository'] not in visited:
                    validate_source(source, data / 'snapshots' / relative(task['repository'].replace('/', '--')),
                                    'training')
                    visited.add(task['repository'])
                selected.append(prepare_case(root, stage, row, number, proofs[row['request_sha256']], task,
                                             source, profile, tokenizer, case_design))
            normalizer_files = ('experiments/023-evaluation-stability/normalize.py',
                                'experiments/024-feedback-noise/normalize.py',
                                'experiments/024-feedback-noise/_feedback024.py')
            inputs = (DESIGN, Path(__file__), root / INTAKE, root / REPLAY, root / INITIAL, root / RUNTIME,
                      root / BASELINE, root / BASELINE_REVIEW, root / ORIGINAL_PROTOCOL, source_contract,
                      student_path, *(root / name for name in normalizer_files))
            frozen = {str(path.relative_to(root)): sha256(path) for path in inputs}
            frozen.update({str(path.relative_to(root)): sha256(path) for path in (root / 'one_gpu_lm').rglob('*.py')})
            frozen.update({str((plan.parent / path.relative_to(stage)).relative_to(root)): sha256(path)
                           for path in stage.rglob('*.json') if 'workspace' not in path.relative_to(stage).parts})
            atomic_json(stage / plan.name, {'schema_version': 1, 'design_sha256': sha256(DESIGN),
                        'training_only': True, 'final_test_opened': False, 'rows': selected, 'frozen_files': frozen,
                        'agent_policy': parent['agent_policy'], 'output': str(OUTPUT)})
            stage.rename(plan.parent)
    additions = [row['added_input_tokens'] for row in selected]
    return {'protocol_sha256': sha256(plan), 'design_sha256': sha256(DESIGN), 'states': len(selected),
            'families': len(visited), 'added_input_tokens_range': [min(additions), max(additions)],
            'physical_rollouts': 2 * len(selected), 'training_only': True, 'final_test_opened': False}


def registered(args):
    root, path = args.root, args.plan
    plan, design, receipt = read(path), read(DESIGN), read(args.publication)
    require(plan['design_sha256'] == sha256(DESIGN) and receipt['protocol_sha256'] == sha256(path)
            and receipt['anonymous_files_verified'] is True and receipt['ci_conclusion'] == 'success'
            and plan['training_only'] is True and plan['final_test_opened'] is False
            and plan['output'] == str(OUTPUT)
            and all(sha256(root / relative(name)) == digest for name, digest in plan['frozen_files'].items()),
            'Run only the original verified training registration')
    data, tasks, sources = reviewed_corpus(root, read(root / INITIAL))
    cases = {}
    reference = {'model_id': design['model_id'], 'revision': design['base_revision'],
                 'evaluation_profile_sha256': design['profile_sha256'], 'agent_policy': plan['agent_policy']}
    for key in ('environment', 'evaluation_profile', 'model_manifest'):
        reference[key + '_path'] = str((path.parent / (key + '.json')).relative_to(root))
    require([row['number'] for row in plan['rows']] == list(range(1, design['training_states'] + 1)),
            'Use every registered training state in its original order')
    for row in plan['rows']:
        snapshot = path.parent / relative(row['snapshot'])
        prefix, manifest = read(snapshot / 'prefix.json'), read(snapshot / 'manifest.json')
        task = tasks[row['task_id']]
        source = sources[task['repository']]
        require(task['split'] == source['split'] == 'training' and task['family'] == row['family'] == source['family']
                and prefix['messages']['guided'] == with_guide(prefix['messages']['control'], design['guide'])
                and inventory(snapshot / 'workspace') == manifest, 'The frozen training state or paired guide changed')
        branches = {}
        for arm in ARMS:
            opening = prefix['openings'][arm]
            messages = prefix['messages'][arm]
            proof = {'prefix_sha256': sha256(snapshot / 'prefix.json'), 'arms': {'control': opening},
                     'request_messages_sha256': message_hash(messages),
                     'packed_messages_sha256': opening['packed_messages_sha256']}
            branches[arm] = DecisionCase(task, source, snapshot, manifest, prefix, proof, messages,
                                         dict(reference, evaluation_budget=prefix['budget']))
        cases[row['number']] = branches
    return plan, design, reference, data, cases


def collect(args):
    plan, design, reference, data, cases = registered(args)
    root, out = args.root, args.root / OUTPUT
    runtime = read(root / RUNTIME)
    runner = Sandbox(read(args.plan.parent / 'sandbox.json'))
    with single_writer(gpu_lock_path()):
        require(not out.exists(), 'Preserve the original GPU attempt')
        (out / 'private').mkdir(parents=True)
        start = time.time()
        state = {'phase': 'loading', 'complete': False, 'protocol_sha256': sha256(args.plan),
                 'started_epoch': start, 'deadline_epoch': start + design['pipeline_seconds'],
                 'runs': [], 'backend_lifecycles': [], 'expected_rollouts': 2 * len(cases),
                 'training_only': True, 'final_test_opened': False, 'optimizer_updates': 0}

        def save():
            atomic_json(out / 'controller.json', state)

        save()
        telemetry = Telemetry(out / 'telemetry.csv', 1)
        backend = JournalBackend(root, out, runtime, reference, state['backend_lifecycles'], save)
        telemetry.start()
        try:
            profile = read(root / reference['evaluation_profile_path'])
            atomic_json(out / 'model-verification.json',
                        verify_model(runtime['model'], root / reference['model_manifest_path'], profile))
            recovery = design['recovery_seconds_per_request']
            for number, arm, run_id in schedule(len(cases)):
                case = cases[number][arm]
                reserved = case.budget['max_task_seconds'] + recovery + 2 * case.budget['max_tool_seconds']
                require(time.time() + reserved + (600 if backend.role is None else 0) < state['deadline_epoch'],
                        'The frozen deadline must leave a complete actor and grader budget')
                socket = backend.ensure('base', state['deadline_epoch'])
                state.update(phase='collecting', active_run=run_id)
                save()
                actor = out / 'private/tasks' / run_id
                trace = initialize(actor, case, 'control')
                advance(root, case.task, case.protocol, actor, trace, socket, runner, 'control',
                        journal_path=backend.journal_path, recovery_seconds=recovery)
                backend.check_exclusive()
                result = finish(data, case.task, case.source, case.protocol, actor, trace, runner)
                result.update(run_id=run_id, arm='control', experiment_arm=arm, number=number,
                              timeout_call=trace.get('timeout_call'),
                              accounting_drain_seconds=trace.get('accounting_drain_seconds', 0.0),
                              complete_success_within_budget=trace['eligible_verified_trajectory'])
                atomic_json(actor / 'result.json', result)
                state['runs'].append(result)
                save()
            state.update(phase='completed', complete=True, active_run=None)
        except BaseException as error:
            state.update(phase='failed', error_type=type(error).__name__)
            raise
        finally:
            try:
                backend.stop()
            finally:
                state.update(finished_epoch=time.time(), backend_stopped=backend.process is None,
                             telemetry=telemetry.stop(), finished_utc=utc())
                save()


def supervision(case, guide, directory, trace, profile, student, tokenizer):
    final = trace['final_source_sha256']
    edits = [step for step in trace['steps'] if step.get('source_after', step['source_before']) != step['source_before']]
    if not edits or edits[-1]['source_after'] != final or final == trace['original_source_sha256']:
        return [], 'no_verified_edit'
    last_edit = edits[-1]
    later = trace['steps'][last_edit['call']:]
    verified = next((step for step in later if step.get('command') == case.task['regression_command']
                     and step['source_before'] == step.get('source_after') == final
                     and target_is_executed(step) and step['observation']['returncode'] == 0), None)
    if verified is None or not later or later[-1].get('command') != SUBMIT or later[-1]['call'] <= verified['call']:
        return [], 'no_verified_submission'
    targets = (last_edit, verified, later[-1])
    if any(not target_is_executed(step) or step['observation']['returncode'] != 0
           or not step['protected_before'] or not step['protected_after'] for step in targets):
        return [], 'unexecuted_target'
    runtime = ReplayRuntime(None, None, tokenizer, student, profile, case.protocol['agent_policy'], case.budget)
    selected = []
    opening = with_guide(case.messages, guide)
    for step, kind in zip(targets, ('edit', 'regression', 'submit')):
        request_path = directory / f"request-{step['call']:03d}.json"
        messages = read(request_path)['messages']
        require(messages[:len(opening)] == opening, 'Guided request lost the registered opening')
        messages[0] = deepcopy(case.messages[0])
        target = step['generation']['text']
        try:
            packed, encoding = runtime.target(messages, target)
        except ValueError:
            return [], 'encoding_budget'
        row = {'kind': kind, 'messages': packed, 'target': target, 'encoding': encoding,
               'provenance': {'split': 'training', 'task_id': case.task['id'], 'family': case.task['family'],
                              'call': step['call'], 'guided_request_sha256': sha256(request_path),
                              'trajectory_sha256': sha256(directory / 'trajectory.json'), 'guide_removed': True}}
        serialized = json.dumps(row, ensure_ascii=False)
        if guide in serialized or any(re.search(pattern, serialized) for pattern in SECRET_SHAPES):
            return [], 'unapproved_target_content'
        selected.append(row)
    return selected, 'qualified'


def audit(args):
    plan, design, reference, data, cases = registered(args)
    root, out = args.root, args.root / OUTPUT
    controller, target = out / 'controller.json', out / 'audit.json'
    state = read(controller)
    expected = list(schedule(len(cases)))
    require(not target.exists() and state['protocol_sha256'] == sha256(args.plan)
            and state['training_only'] is True and state['final_test_opened'] is False
            and state['complete'] and state['phase'] == 'completed' and state['backend_stopped']
            and state['started_epoch'] < state['finished_epoch'] <= state['deadline_epoch']
            and [record['run_id'] for record in state['runs']] == [item[2] for item in expected],
            'An independent audit requires all 26 registered executions')
    profile = read(root / reference['evaluation_profile_path'])
    student = read(root / relative(read(root / BASELINE)['profile_path']))
    tokenizer = load_tokenizer(root)
    normalizer = normalization(root)
    loads = state['backend_lifecycles']
    require(len(loads) == 1 and loads[0]['role'] == 'base', 'Use one pinned base load for both arms')
    load = loads[0]
    journal = RequestJournal(out / 'private/backends/0001-base/requests')
    seen, groups = set(), {number: {} for number in cases}
    for (number, arm, name), result in zip(expected, state['runs'], strict=True):
        case = cases[number][arm]
        actor = out / 'private/tasks' / name
        require((result['number'], result['experiment_arm']) == (number, arm), 'Physical actor pairing changed')
        cell, trace, requests = check_arm(root, data, out, actor, case,
                                         {'task_id': case.task['id'], 'arm': 'control'}, result,
                                         journal, profile, tokenizer, normalizer)
        require(not seen.intersection(requests) and load['ready_epoch'] <= trace['started_epoch']
                <= trace['finished_epoch'] <= load['stopped_epoch'], 'Generation receipts overlap or timing changed')
        seen.update(requests)
        groups[number][arm] = (cell, trace, actor)
    require({path.name for path in journal.directory.iterdir()} == seen
            and len(seen) == sum(record['attempted_calls'] for record in state['runs']),
            'Every registered request must have exactly one durable receipt')
    pairs, labels, families = [], [], set()
    for number, group in groups.items():
        control, guided = (group[arm][0]['verified_and_submitted'] for arm in ARMS)
        if guided and not control:
            _, trace, actor = group['guided']
            emitted, reason = supervision(cases[number]['control'], design['guide'], actor, trace,
                                          profile, student, tokenizer)
            require(len(emitted) in (0, 3), 'Episode admission is atomic')
            if emitted:
                labels.extend(emitted)
                families.add(cases[number]['control'].task['family'])
        else:
            emitted, reason = [], 'control_only' if control and not guided else 'same_outcome'
        pairs.append({'pair': number, 'admission': reason, 'qualified_targets': len(emitted),
                      'arms': {arm: {key: group[arm][0][key] for key in CELL_FIELDS} for arm in ARMS}})
    data_path = out / 'private/suffix-supervision.json'
    if data_path.exists():
        require(read(data_path) == labels, 'Preserve the first audited supervision bytes')
    else:
        atomic_json(data_path, labels)
    atomic_json(target, {'schema_version': 1, 'protocol_sha256': sha256(args.plan),
                'controller_sha256': sha256(controller), 'passed': True, 'pairs': pairs,
                'qualified_families': len(families), 'verified_requests': len(seen),
                'private_supervision_sha256': sha256(data_path), 'training_only': True,
                'final_test_opened': False, 'optimizer_updates': 0, 'weights_public': False})


def capture(args):
    root, out = args.root, args.root / OUTPUT
    audit_path, controller = out / 'audit.json', out / 'controller.json'
    audited, state = read(audit_path), read(controller)
    receipt = read(args.publication)
    require(audited['passed'] is audited['training_only'] is True and audited['final_test_opened'] is False
            and receipt['anonymous_files_verified'] is True and receipt['ci_conclusion'] == 'success'
            and audited['controller_sha256'] == sha256(controller)
            and audited['protocol_sha256'] == sha256(args.plan) == receipt['protocol_sha256']
            and audited['private_supervision_sha256'] == sha256(out / 'private/suffix-supervision.json'),
            'Publish only the complete audited projection')
    gpu = hardware(out / 'telemetry.csv', state)
    require(gpu['samples'] == state['telemetry']['samples'], 'Telemetry count differs from the collector')
    return {key: audited[key] for key in ('protocol_sha256', 'controller_sha256', 'passed', 'pairs',
                                        'qualified_families', 'verified_requests', 'training_only',
                                        'final_test_opened', 'optimizer_updates', 'weights_public')} | {
        'audit_sha256': sha256(audit_path), 'registration_commit': receipt['commit'], 'hardware': gpu,
        'started_epoch': state['started_epoch'], 'finished_epoch': state['finished_epoch']}


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'audit', 'capture'))
    for name in ('root', 'plan'):
        parser.add_argument('--' + name, type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument('--publication', type=Path)
    args = parser.parse_args()
    if args.action != 'freeze' and args.publication is None:
        parser.error('Run, audit and capture require the verified public registration')
    result = {'freeze': freeze, 'run': collect, 'audit': audit, 'capture': capture}[args.action](args)
    if result is not None:
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
