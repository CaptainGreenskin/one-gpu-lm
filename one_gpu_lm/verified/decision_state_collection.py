"""Collect paired native-tool continuations from audited training states."""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from shutil import copyfile
import time

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.runstate import atomic_json, gpu_lock_path, single_writer, utc
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.verify import verify_model
from .correction import schedule
from .correction_rollout import advance, finish
from .decision_policy import REASONING_VERSION
from .first_edit_continuation import message_hash
from .recovery_state import copy_tree, inventory
from .resilient_backend_control import JournalBackend
from .sandbox import Sandbox
from .submission import SubmissionGuard
from .tasks import files, relative


def read(path):
    return json.loads(Path(path).read_bytes())


def require(valid, reason):
    if not valid:
        raise ValueError(reason)


def history_and_budget(prefix, reference):
    messages = deepcopy(prefix['messages'])
    budget = dict(reference['evaluation_budget'])
    budget['max_model_calls'] -= prefix['historical_calls']
    budget['max_total_generated_tokens'] -= prefix['historical_generated_tokens']
    require(budget['max_model_calls'] == prefix['remaining_calls'] > 0
            and budget['max_total_generated_tokens'] == prefix['remaining_generated_tokens'] > 0,
            'Continuation call or generation limits differ from the replay')
    calls = budget['max_model_calls']
    tokens = budget['max_total_generated_tokens']
    per_call = budget['max_generated_tokens_per_call']
    seconds = budget['max_task_seconds']
    messages[0]['content'] += (
        f'\nThe repository is restored to the state reached in the visible conversation. '
        f'For this continuation the earlier limits are superseded by {calls} model calls, '
        f'{tokens} generated tokens in total, at most {per_call} generated tokens per call '
        f'and {seconds} seconds of new actor time. Each model reply counts as a call. '
        'Continue the task from the tool evidence already shown.'
    )
    return messages, budget


@dataclass
class DecisionCase:
    task: dict
    source: dict
    snapshot: Path
    manifest: dict
    prefix: dict
    proof: dict
    messages: list
    protocol: dict

    @property
    def kind(self):
        return self.prefix['kind']

    @property
    def budget(self):
        return self.protocol['evaluation_budget']

    @property
    def initial_verified_source(self):
        return self.prefix['verified_source']


def load_cases(root, plan):
    for name, expected in plan['frozen_files'].items():
        require(sha256(root / relative(name)) == expected, 'Frozen collection input changed')
    design = read(root / relative(plan['design_path']))
    reference_path = root / relative(design['reference_path'])
    reference = read(reference_path)
    provenance = read(root / relative(design['collection_provenance_path']))
    require(design['final_test_opened'] is False and provenance['final_test_opened'] is False
            and reference['downstream']['final_test_opened'] is False
            and (reference['model_id'], reference['revision']) == (design['model_id'], design['base_revision'])
            and reference['reasoning_comparison']['arms'] == design['reasoning_modes']
            and reference['continuation_policy'] == REASONING_VERSION
            and reference['evaluation_budget'] == design['total_budget']
            and reference['agent_policy']['format'] == 'native_roundtrip',
            'Registered collection model, modes or budget differs')

    replay_path = root / relative(plan['snapshot_report_path'])
    replay = read(replay_path)
    inputs = read(root / relative(plan['tokenizer_screen_path']))
    require(sha256(replay_path) == inputs['replay_report_sha256']
            and sha256(reference_path) == inputs['reference_sha256']
            and inputs['profile_sha256'] == reference['evaluation_profile_sha256']
            and (inputs['model_id'], inputs['base_revision']) == (reference['model_id'], reference['revision'])
            and replay['new_generations'] == inputs['new_generations'] == inputs['optimizer_updates'] == 0
            and replay['final_test_opened'] is inputs['final_test_opened'] is False,
            'CPU replay or paired tokenizer proof differs')
    proof = {r['task_id']: r for r in inputs['rows']}
    rows = replay['rows']
    ids = [r['task_id'] for r in rows]
    require(len(ids) == len(proof) == len(inputs['rows']) == len(set(ids)) == sum(design['training_states'].values())
            and set(ids) == set(proof) and plan['schedule'] == schedule(ids)
            and len(plan['schedule']) == design['expected_rollouts'],
            'Training state count or scheduled pairing differs')
    data = root / relative(reference['data_directory'])
    require(provenance['data_directory'] == reference['data_directory']
            and all(sha256(data / name) == provenance['frozen_files'][str(relative(reference['data_directory']) / name)]
                    for name in ('tasks.json', 'sources.json')),
            'Training task or source inventory differs from the admitted collection')
    tasks = {t['id']: t for t in read(data / 'tasks.json')['tasks'] if t['id'] in proof}
    sources = {s['repository']: s for s in read(data / 'sources.json')['sources']}
    cases = {}
    for row in rows:
        name = row['task_id']
        task = tasks[name]
        source = sources[task['repository']]
        verified = proof[name]
        snapshot = replay_path.parent / relative(row['snapshot'])
        hashes = row['files_sha256']
        require(row['status'] == 'admitted' and snapshot.parent == replay_path.parent
                and not snapshot.is_symlink()
                and set(hashes) == {'prefix.json', 'manifest.json', 'before.json', 'replay.json'}
                and all(sha256(snapshot / file) == expected for file, expected in hashes.items())
                and task['split'] == source['split'] == 'training'
                and source['approved_for']['training'] is True
                and source['notice_review_complete'] is source['lineage_review_complete'] is True
                and row['family'] == verified['family'] == task['family'] == source['family']
                and row['language'] == verified['language'] == task['language'],
                'A registered training snapshot or source approval differs')
        prefix = read(snapshot / 'prefix.json')
        manifest = read(snapshot / 'manifest.json')
        workspace = snapshot / 'workspace'
        messages, budget = history_and_budget(prefix, reference)
        original = prefix['original_source_sha256']
        current = prefix['decision_source_sha256']
        passing = prefix['verified_source']
        kind = prefix['kind']
        require(prefix['final_test_opened'] is False
                and set(task['editable_files']) == set(current) == set(original)
                and (kind == 'early_edit' and passing is None and current == original
                     and prefix['historical_calls'] in (1, 2) and row['standalone_returncode'] != 0
                     or kind == 'verified_finish' and passing == current and current != original
                     and all(h is not None for h in current.values()) and row['standalone_returncode'] == 0)
                and current == SubmissionGuard.source_state(workspace, task)
                and inventory(workspace) == manifest and not set(task['hidden_checks']).intersection(manifest)
                and verified['kind'] == row['kind'] == kind and verified['budget'] == budget
                and verified['prefix_sha256'] == hashes['prefix.json']
                and verified['request_messages_sha256'] == message_hash(messages)
                and {arm: p['mode'] for arm, p in verified['arms'].items()} == design['reasoning_modes'],
                'The restored source, verification, context or remaining budget differs')
        cases[name] = DecisionCase(task, source, snapshot, manifest, prefix, verified, messages,
                                   {**reference, 'evaluation_budget': budget})
    require(Counter(c.kind for c in cases.values()) == design['training_states']
            and len({c.task['family'] for c in cases.values()}) == design['training_families']
            and 2 * sum(c.budget['max_model_calls'] for c in cases.values()) == design['max_new_calls_across_both_arms']
            and 2 * sum(c.budget['max_total_generated_tokens'] for c in cases.values()) ==
            design['max_new_tokens_across_both_arms'], 'Cohort or global paired ceilings differ')
    return design, reference, data, cases


def restoration_receipt(case):
    return {'prefix_sha256': case.proof['prefix_sha256'],
            'manifest_sha256': sha256(case.snapshot / 'manifest.json'),
            'initial_messages_sha256': case.proof['request_messages_sha256']}


def initialize(directory, case, arm):
    directory.mkdir(parents=True)
    workspace = directory / 'workspace'
    copy_tree(case.snapshot / 'workspace', workspace, case.manifest)
    copyfile(case.snapshot / 'before.json', directory / 'before.json')
    require(SubmissionGuard.source_state(workspace, case.task) == case.prefix['decision_source_sha256']
            and not set(case.task['hidden_checks']).intersection(files(workspace)),
            'Actor restoration differs from the audited decision state')
    write_json(directory / 'restored-state.json', restoration_receipt(case))
    task = case.task
    state = {'schema_version': 1, 'id': task['id'], 'repository': task['repository'], 'family': task['family'],
             'split': 'training', 'branch': arm, 'producer_kind': 'model', 'started_epoch': time.time(),
             'complete': False, 'boundary': 'before_generation', 'messages': deepcopy(case.messages),
             'steps': [], 'backend_identities': {}, 'attempted_calls': 0, 'generated_tokens': 0,
             'actor_seconds': 0.0, 'submitted': False, 'terminal_reason': None,
             'verified_source': deepcopy(case.initial_verified_source),
             'original_source_sha256': case.prefix['original_source_sha256']}
    atomic_json(directory / 'trajectory.json', state)
    return state


def run(root, protocol_path):
    plan = read(protocol_path)
    design, reference, data, cases = load_cases(root, plan)
    public = read(root / relative(plan['publication_receipt']))
    protocol_hash = sha256(protocol_path)
    require(public['anonymous_files_verified'] is True and public['ci_conclusion'] == 'success'
            and public['protocol_sha256'] == protocol_hash, 'Verified public registration required')
    runtime = read(root / 'private/parent-runtime.json')
    runner = Sandbox(read(root / 'private/task-runtime.json'), timeout=design['total_budget']['max_tool_seconds'])
    out = root / relative(plan['run_directory'])
    with single_writer(gpu_lock_path()):
        (out / 'private').mkdir(parents=True)
        state = {'schema_version': 1, 'stage': 'verifying_model', 'complete': False, 'started_epoch': time.time(),
                 'started_utc': utc(), 'protocol_sha256': protocol_hash, 'publication_commit': public['commit'],
                 'runs': [], 'backend_lifecycles': [], 'expected_runs': len(plan['schedule']),
                 'final_test_opened': False, 'optimizer_updates': 0}

        def save():
            atomic_json(out / 'controller.json', state)

        save()
        telemetry = Telemetry(out / 'telemetry.csv', 1)
        backend = JournalBackend(root, out, runtime, reference, state['backend_lifecycles'], save)
        recovery = design['recovery_seconds_per_rollout']
        # Each actor gets its full time, response recovery and a separate grader.
        actor = design['total_budget']['max_task_seconds']
        reserved = actor + recovery + 2 * design['total_budget']['max_tool_seconds']
        deadline = state['started_epoch'] + 900 + len(plan['schedule']) * reserved
        telemetry.start()
        try:
            profile = read(root / relative(reference['evaluation_profile_path']))
            write_json(out / 'model-verification.json',
                       verify_model(runtime['model'], root / relative(reference['model_manifest_path']), profile))
            for ordinal, item in enumerate(plan['schedule'], 1):
                require(time.time() + reserved + (600 if backend.role is None else 0) <= deadline,
                        'Pipeline has no time for the next complete registered attempt')
                case = cases[item['task_id']]
                socket = backend.ensure('base', deadline)
                state.update(stage='collecting', current_ordinal=ordinal, current_kind=case.kind)
                save()
                arm = item['arm']
                directory = out / 'private' / 'tasks' / relative(item['run_id'])
                trace = initialize(directory, case, arm)
                advance(root, case.task, case.protocol, directory, trace, socket, runner, arm,
                        journal_path=backend.journal_path, recovery_seconds=recovery,
                        initial_verified_source=case.initial_verified_source)
                backend.check_exclusive()
                result = finish(data, case.task, case.source, case.protocol, directory, trace, runner)
                result.update(run_id=item['run_id'], arm=arm, backend_load=backend.number,
                              timeout_call=trace.get('timeout_call'),
                              accounting_drain_seconds=trace.get('accounting_drain_seconds', 0.0),
                              complete_success_within_budget=trace['eligible_verified_trajectory'])
                write_json(directory / 'result.json', result)
                state['runs'].append(result)
                save()
            state.update(stage='completed', complete=True)
        except BaseException as error:
            state.update(stage='failed', error_type=type(error).__name__)
            raise
        finally:
            try:
                backend.stop()
            finally:
                state.update(telemetry=telemetry.stop(), finished_epoch=time.time(), finished_utc=utc(),
                             backend_stopped=backend.process is None,
                             frozen_files_unchanged=all(sha256(root / relative(name)) == expected
                                                        for name, expected in plan['frozen_files'].items()))
                save()
    return state
