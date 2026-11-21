"""Independently replay successful training continuations with deployment feedback."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import format_feedback, prepare_context
from one_gpu_lm.common.reasoning import action_text
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import identity_digest
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.training_encoding import encode_training_example

from .continuation_training_data import read
from .decision_state_audit import packed_history
from .decision_state_collection import load_cases, require
from .failure_states import protected, source_state
from .guard_audit import target_is_executed
from .historic_policy import feedback, initial_messages
from .historic_rollout import normalization
from .sandbox import Sandbox
from .submission import SUBMIT
from .tasks import grade, prepare, relative


COLLECTION_PROTOCOL = Path('private/protocol.json')
COLLECTION_JOB = Path('private/decision-043-job/state.json')
ORIGINAL_PROTOCOL = Path('experiments/025-verified-distillation/collection-protocol.json')
TARGET_KIND = 'deployment_replayed_continuation'


@dataclass(frozen=True)
class ReplayRuntime:
    sandbox: object
    normalizer: object
    tokenizer: object
    student: dict
    teacher: dict
    policy: dict
    budget: dict

    def feedback(self, output):
        return feedback(output, self.normalizer, self.budget['max_tool_output_chars'])[0]

    def target(self, messages, text):
        selected = []

        def encode(items):
            selected[:] = items
            return encode_prefix(self.tokenizer, items, self.student['chat_template_kwargs'], self.policy['format'])[0]

        prefix, _ = prepare_context(messages, encode, self.budget['max_input_tokens'], self.policy['context'])
        encoded = encode_training_example({'messages': selected, 'target': text}, self.tokenizer, self.student)
        require(encoded['ids'][:encoded['prefix_tokens']] == prefix
                and 0 < encoded['supervised_tokens'] <= self.budget['max_generated_tokens_per_call']
                and len(encoded['ids']) <= self.student['training']['max_sequence_length'],
                'replayed_target_exceeds_deployment_budget')
        return selected, {'tokens': len(encoded['ids']), 'target_tokens': encoded['supervised_tokens'],
                          'encoding_sha256': identity_digest(encoded['ids'])}


def recorded_action(step, text, directory):
    """Classify an audited response before any sandbox operation."""
    if not step['action_format_valid']:
        require('observation' not in step and 'source_after' not in step, 'malformed_action_was_executed')
        return None, None, 'format_feedback'
    action = parse_action(text)
    require(action['command'] == step['command'], 'tool_command_does_not_match_receipt')
    raw = read(directory / f"tool-{step['call']:03d}.json")
    require({k: v for k, v in raw.items() if k != 'output'} == step['observation'], 'raw_tool_receipt_changed')
    if not target_is_executed(step):
        require(raw['executed'] is False and raw['reason'] in
                ('correction_requires_regression', 'submission_requires_verification')
                and step['source_after'] == step['source_before'], 'unverified_collection_rejection')
        return None, None, 'collection_rejection_removed'
    return action, raw, None


def advance(runtime, workspace, original, task, directory, step, text, state, messages):
    require(step['source_before'] == state == source_state(workspace, task), 'before_source_changed')
    action, raw, skipped = recorded_action(step, text, directory)
    if skipped:
        if skipped == 'format_feedback':
            messages.extend([{'role': 'assistant', 'content': text, 'extra': {}},
                             {'role': 'user', 'content': format_feedback(runtime.policy)}])
        return state, skipped
    command = action['command']
    actual = runtime.sandbox.execute(workspace, command)
    delivered = runtime.feedback(actual['output'])
    require((actual['returncode'], actual['reason'], delivered) ==
            (raw['returncode'], raw['reason'], runtime.feedback(raw['output'])), 'normalized_tool_feedback_changed')
    after = source_state(workspace, task)
    require(after == step['source_after'] and protected(workspace, original, task), 'after_source_changed')
    messages.extend([{'role': 'assistant', 'content': text, 'extra': {'actions': [action]}},
                     {'role': 'user', 'content': f"Exit code: {actual['returncode']}\n{delivered}",
                      'extra': {'tool': {'returncode': actual['returncode'], 'reason': actual['reason']}}}])
    if after != state:
        kind = 'edit'
    elif command == task['regression_command']:
        kind = 'regression_pass' if actual['returncode'] == 0 else 'regression_fail'
    elif command == SUBMIT and actual['returncode'] == 0:
        kind = 'submit'
    else:
        kind = 'other'
    return after, kind


def replay_case(runtime, snapshot, historical, historical_row, collection, trace, case, work, lineage):
    task = case.task
    prior_path = historical / 'trajectory.json'
    require(sha256(prior_path) == historical_row['trajectory_sha256']
            and historical_row['input_replay_passed'] and historical_row['complete'], 'historical_prefix_receipt_changed')
    prior = read(prior_path)
    cut = case.prefix['historical_calls']
    require(prior['split'] == 'training' and prior['messages'][:2 + 2*cut] == case.prefix['messages']
            and prior['original_source_sha256'] == case.prefix['original_source_sha256'], 'historical_opening_changed')

    probe = work / 'opening'
    original = prepare(snapshot, probe, case.source, task)
    state = source_state(probe, task)
    failure = runtime.sandbox.execute(probe, task['regression_command'])
    require(state == case.prefix['original_source_sha256'] == source_state(probe, task)
            and protected(probe, original, task) and failure['reason'] == 'runtime_error'
            and failure['returncode'] != 0, 'opening_regression_failure_did_not_reproduce')
    opening = ('\n\nBefore any source changes, the harness ran the declared public regression command. '
               f"Exit code: {failure['returncode']}. Run the declared check again after editing."
               '\nPublic regression output follows. Treat it as diagnostic data.\n' + runtime.feedback(failure['output']))
    messages = initial_messages(dict(task, description=task['description'] + opening), runtime.policy, runtime.budget)
    actor = work / 'actor'
    original = prepare(snapshot, actor, case.source, task)
    total = 0
    cap = runtime.budget['max_generated_tokens_per_call']
    counts = Counter()
    for step in prior['steps'][:cut]:
        state, kind = advance(runtime, actor, original, task, historical, step, step['generation']['text'], state, messages)
        require(kind != 'collection_rejection_removed' and state == case.prefix['decision_source_sha256'],
                'unexpected_historical_opening_action')
        cost = step['generation']['generated_tokens']
        require(0 < cost <= cap, 'historical_opening_exceeds_call_budget')
        total += cost
    require(total == case.prefix['historical_generated_tokens'], 'historical_token_receipt_changed')

    learned = []
    verified = None
    episode = identity_digest([TARGET_KIND, task['id'], lineage['trajectory_sha256']])
    for step in trace['steps']:
        text = action_text(step['generation'], case.protocol)
        executed = target_is_executed(step)
        if executed:
            packed_history(collection, step, runtime.teacher, runtime.tokenizer, runtime.policy)
            selected, encoding = runtime.target(messages, text)
            total += encoding['target_tokens']
        elif not step['action_format_valid']:
            cost = step['generation']['generated_tokens']
            require(0 < cost <= cap, 'format_error_exceeds_call_budget')
            total += cost
        state, kind = advance(runtime, actor, original, task, collection, step, text, state, messages)
        counts[kind] += 1
        if kind in ('edit', 'regression_fail'):
            verified = None
        elif kind == 'regression_pass':
            verified = state
        if executed:
            learned.append({'id': identity_digest([episode, step['call']]), 'episode_id': episode,
                            'messages': selected, 'target': text, 'encoding_receipt': encoding,
                            'provenance': dict(lineage, split='train', producer_kind='model', executed=True,
                                               task_id=task['id'], repository=task['repository'], family=task['family'],
                                               call=step['call'], kind=kind, target_transform=TARGET_KIND)})
    require(counts['edit'] > 0 and counts['submit'] == 1 and learned[-1]['provenance']['kind'] == 'submit'
            and state == verified == trace['final_source_sha256'] and state != case.prefix['original_source_sha256']
            and 0 < (len(messages) - 2)//2 <= runtime.budget['max_model_calls']
            and total <= runtime.budget['max_total_generated_tokens'], 'replay_did_not_verify_and_submit_within_budget')
    result = grade(snapshot, work/'graded', case.source, task, actor, original, runtime.sandbox.execute)
    require(result['passed'] is True and result['source_sha256'] == state, 'independent_training_grade_failed')
    return learned, {'counts': dict(counts), 'replayed_calls': (len(messages) - 2)//2,
                     'replayed_generated_token_accounting': total, 'graded_source_sha256': identity_digest(state),
                     'historical_prefix_sha256': sha256(prior_path)}


def collection_inputs(root, design):
    collection = design['collection']
    path = root / COLLECTION_PROTOCOL
    require(sha256(path) == collection['protocol_sha256'], '043 collection protocol changed')
    plan = read(path)
    out = root / relative(plan['run_directory'])
    audit_path, controller_path = out/'audit.json', out/'controller.json'
    require(sha256(audit_path) == collection['audit_sha256'], '043 collection audit changed')
    audit, controller, job = read(audit_path), read(controller_path), read(root / COLLECTION_JOB)
    require(audit['passed'] and audit['training_only'] and not audit['optimizer_updates']
            and controller['complete'] and controller['backend_stopped'] and controller['frozen_files_unchanged']
            and audit['protocol_sha256'] == controller['protocol_sha256'] == job['protocol_sha256'] == sha256(path)
            and audit['controller_sha256'] == sha256(controller_path)
            and job['phase'] == 'audited' and job['audit_sha256'] == sha256(audit_path)
            and audit['audited_attempts'] == len(controller['runs']) == len(plan['schedule'])
            and audit['journal_requests_verified'] == sum(row['attempted_calls'] for row in controller['runs'])
            and all(item['final_test_opened'] is False for item in (audit, controller, job)),
            'Require a stopped and completely audited training collection')
    return plan, out, audit, controller


def supplements(root, collection, baseline, tokenizer, design):
    """Replay and export one complete block for every eligible new training task."""
    plan, out, audit, controller = collection_inputs(collection, design)
    specification, reference, data, cases = load_cases(collection, plan)
    historical_plan = root / ORIGINAL_PROTOCOL
    historical = root / relative(read(historical_plan)['run_directory'])
    historical_audit = historical / 'audit.json'
    require(sha256(historical_plan) == design['collection']['historical_protocol_sha256']
            and sha256(historical_audit) == design['collection']['historical_audit_sha256'],
            'Historical opening audit changed')
    source_hash = sha256(data/'sources.json')
    require(source_hash == baseline.baseline['source_manifest_sha256']
            and sha256(data/'tasks.json') == baseline.review['task_manifest_sha256']
            and (baseline.profile['model_id'], baseline.profile['revision']) == (reference['model_id'], reference['revision'])
            and read(collection/relative(specification['student_profile_path'])) == baseline.profile,
            'Replayed and retained records did not use the same reviewed sources and student profile')
    teacher = read(collection/relative(reference['evaluation_profile_path']))
    runtime = ReplayRuntime(Sandbox(read(collection/'private/task-runtime.json'), design['actor_budget']['max_tool_seconds']),
                            normalization(root), tokenizer, baseline.profile, teacher, design['agent_policy'], design['actor_budget'])
    prior = {row['id']: row for row in read(historical_audit)['rows']}
    runs = {row['run_id']: row for row in controller['runs']}
    retained_tasks = {row['provenance']['task_id'] for row in baseline.raw}
    counts, records, outcomes = Counter(), [], []
    require(len(cases)*2 == audit['audited_attempts'] and len(audit['pairs']) == len(cases), 'Incomplete paired audit')
    with TemporaryDirectory(prefix='replay-', dir=root/'private') as temporary:
        for (task_id, case), pair in zip(cases.items(), audit['pairs']):
            if case.kind != 'early_edit' or not pair['arms']['correction']['verified_and_submitted']:
                continue
            counts['positive_sequences'] += 1
            name = task_id + '--correction'
            where = out / 'private/tasks' / relative(name)
            trajectory = where / 'trajectory.json'
            require(sha256(trajectory) == runs[name]['trajectory_sha256'], 'Successful trajectory changed since audit')
            trace = read(trajectory)
            require(trace['split'] == 'training' and trace['eligible_verified_trajectory'] and trace['submitted'],
                    'Expected an independently verified training trajectory')
            lineage = {'trajectory_sha256': sha256(trajectory), 'source_manifest_sha256': source_hash,
                       'collection_audit_sha256': design['collection']['audit_sha256']}
            source = data/'snapshots'/relative(case.source['repository'].replace('/', '--'))
            work = Path(temporary)/str(pair['pair'])
            work.mkdir()
            try:
                selected, receipt = replay_case(runtime, source, historical/'private/tasks'/relative(task_id), prior[task_id],
                                                where, trace, case, work, lineage)
            except ValueError as error:
                counts['replay_rejected'] += 1
                outcomes.append({'pair': pair['pair'], 'selected': False, 'replay_passed': False, 'reason': str(error)})
                continue
            counts['replayed_sequences'] += 1
            counts['replayed_targets'] += len(selected)
            carried = task_id in retained_tasks
            counts['already_in_retained_training'] += carried
            if not carried:
                records.extend(selected)
            outcomes.append({'pair': pair['pair'], 'selected': not carried, 'replay_passed': True,
                             'records': len(selected), **receipt})
    require(counts['positive_sequences'] == design['collection']['positive_sequences'], 'Positive cohort did not reproduce')
    require(bool(records), 'No novel complete verified training continuations')
    return records, {'counts': dict(counts), 'outcomes': outcomes, 'new_model_generations': 0, 'final_test_opened': False}
