"""Run a paired, training-only continuation from a verified unchanged source."""
from copy import deepcopy
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
from .decision_policy import extra_initial
from .recovery_state import copy_tree
from .resilient_backend_control import JournalBackend
from .sandbox import Sandbox
from .tasks import digest, relative


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(valid, message):
    if not valid:
        raise ValueError(message)


def message_hash(messages):
    return digest(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode())


def initial_context(prefix, registered):
    """Match the CPU-screened prompts and deduct already consumed calls and tokens."""
    history = deepcopy(prefix['messages'])
    budget = dict(registered['evaluation_budget'])
    budget['max_model_calls'] -= prefix['historical_calls']
    budget['max_total_generated_tokens'] -= prefix['historical_generated_tokens']
    require(min(budget['max_model_calls'], budget['max_total_generated_tokens']) > 0,
            'Historical prefix exhausted the planned budget')
    calls = budget['max_model_calls']
    per_call = budget['max_generated_tokens_per_call']
    remaining = budget['max_total_generated_tokens']
    history[0]['content'] += (
        f'\nThe historical prefix has ended. For this continuation its limits are superseded by '
        f'{calls} new tool calls, {per_call} generated tokens per model call, {remaining} new generated '
        f'tokens in total, and {budget["max_task_seconds"]} seconds. All reasoning and invalid '
        'responses count. Finish any thinking and return one complete tool action per response. '
        'Preserve enough calls to run the declared regression and submit after it passes.'
    )
    guidance = extra_initial(registered, 'control')
    require(guidance == extra_initial(registered, 'correction'), 'Continuation guidance differs between arms')
    history[-1]['content'] += (
        '\nThe workspace is restored to the unchanged source from the historical prefix. '
        'The declared visible regression was run separately on a copy of this workspace '
        'and still fails. Its observed output follows.\n' + prefix['standalone_visible_failure'] + guidance
    )
    return history, budget


def load_cases(root, plan):
    """Bind the complete CPU selection, exact prompt hashes and reviewed training sources."""
    for name, value in plan['frozen_files'].items():
        require(sha256(root / relative(name)) == value, 'Frozen continuation input changed')
    ref_path = root / relative(plan['reference_path'])
    reference = read(ref_path)
    screen_path = root / relative(plan['cpu_inputs_path'])
    screen = read(screen_path)
    report_path = root / relative(plan['cpu_snapshot_report_path'])
    report = read(report_path)
    design = read(root / relative(plan['design_path']))
    require(screen['comparison_reference_sha256'] == sha256(ref_path)
            and screen['snapshots_report_sha256'] == sha256(report_path)
            and not screen['final_test_opened'] and screen['new_model_generations'] == 0
            and not report['summary']['final_test_opened'] and report['summary']['new_generations'] == 0
            and sha256(screen_path) == design['cpu_inputs_sha256'],
            'CPU replay, template preflight or public design differs')
    require(all(screen[x] == reference[y] for x, y in (
        ('model_id', 'model_id'), ('base_revision', 'revision'),
        ('teacher_profile_sha256', 'evaluation_profile_sha256'))),
        'The CPU comparison used a different model or template')
    require(set(reference['reasoning_comparison']['arms'].values()) == {'direct', 'low'},
            'Expected closed and low reasoning on the same backend')
    require(reference['evaluation_budget'] == design['total_budget'],
            'Registered continuation budget differs from the public design')
    screened = {r['task_id']: r for r in screen['rows']}
    ids = [r['task_id'] for r in report['rows']]
    require(len(ids) == len(set(ids)) == len(screened) == design['task_count']
            and set(ids) == set(screened) and plan['schedule'] == schedule(ids),
            'CPU-proven state coverage or alternating order differs')
    data = root / relative(reference['data_directory'])
    sources = {s['repository']: s for s in read(data / 'sources.json')['sources']}
    tasks = {t['id']: t for t in read(data / 'tasks.json')['tasks'] if t['id'] in screened}
    require(set(tasks) == set(ids), 'Frozen training tasks differ')
    cases = {}
    for row in report['rows']:
        name = row['task_id']
        task = tasks[name]
        source = sources[task['repository']]
        snapshot = root / relative(row['snapshot'])
        expected_hashes = row['files_sha256']
        require(row['status'] == 'admitted' and snapshot.is_relative_to(report_path.parent)
                and set(expected_hashes) == {'prefix.json', 'manifest.json', 'before.json', 'replay.json'}
                and all(sha256(snapshot / key) == value for key, value in expected_hashes.items())
                and task['split'] == source['split'] == 'training'
                and source['approved_for']['training'] is True,
                'A restored prefix or its training-only admission differs')
        prefix = read(snapshot / 'prefix.json')
        history, budget = initial_context(prefix, reference)
        proof = screened[name]
        require(prefix['historical_calls'] == design['historical_calls']
                and not prefix['final_test_opened'] and not prefix['source_changed']
                and expected_hashes['prefix.json'] == proof['prefix_sha256']
                and message_hash(history) == proof['request_messages_sha256'] and budget == proof['budget']
                and {arm: cell['mode'] for arm, cell in proof['arms'].items()} == reference['reasoning_comparison']['arms'],
                'Restored initial message history or paired budget differs from CPU preflight')
        cases[name] = {'task': task, 'source': source, 'prefix': prefix, 'snapshot': snapshot,
                       'messages': history, 'budget': budget, 'screen': proof,
                       'protocol': {**reference, 'evaluation_budget': budget}}
    return reference, data, cases, design


def initialize(directory, case, arm):
    snapshot = case['snapshot']
    prefix = case['prefix']
    task = case['task']
    directory.mkdir(parents=True)
    copy_tree(snapshot / 'tree', directory / 'workspace', read(snapshot / 'manifest.json'))
    copyfile(snapshot / 'before.json', directory / 'before.json')
    write_json(directory / 'restored-state.json', {
        'prefix_sha256': sha256(snapshot / 'prefix.json'), 'manifest_sha256': sha256(snapshot / 'manifest.json'),
        'initial_messages_sha256': message_hash(case['messages'])})
    state = {'schema_version': 1, 'id': task['id'], 'repository': task['repository'], 'family': task['family'],
             'split': 'training', 'branch': arm, 'producer_kind': 'model', 'started_epoch': time.time(),
             'complete': False, 'boundary': 'before_generation', 'messages': deepcopy(case['messages']),
             'steps': [], 'backend_identities': {}, 'attempted_calls': 0, 'generated_tokens': 0,
             'actor_seconds': 0.0, 'submitted': False, 'terminal_reason': None, 'verified_source': None,
             'original_source_sha256': prefix['original_source_sha256'],
             'historical_prefix_cost': {k: prefix[k] for k in
                 ('historical_calls', 'historical_generated_tokens', 'historical_input_tokens')}}
    atomic_json(directory / 'trajectory.json', state)
    return state


def run(root, protocol_path):
    root, protocol_path = Path(root).resolve(), Path(protocol_path).resolve()
    plan = read(protocol_path)
    out = root / relative(plan['run_directory'])
    reference, data, cases, design = load_cases(root, plan)
    public = read(root / relative(plan['publication_receipt']))
    require(public['anonymous_files_verified'] and public['ci_conclusion'] == 'success'
            and public['protocol_sha256'] == sha256(protocol_path), 'Public registration differs')
    runtime = read(root / 'private/parent-runtime.json')
    sandbox = Sandbox(read(root / 'private/task-runtime.json'), timeout=reference['evaluation_budget']['max_tool_seconds'])
    with single_writer(gpu_lock_path()):
        (out / 'private').mkdir(parents=True)
        state = {'schema_version': 1, 'stage': 'verifying_model', 'complete': False, 'started_epoch': time.time(),
                 'started_utc': utc(), 'protocol_sha256': sha256(protocol_path), 'publication_commit': public['commit'],
                 'runs': [], 'backend_lifecycles': [], 'expected_runs': len(plan['schedule']),
                 'final_test_opened': False, 'optimizer_updates': 0}
        def save():
            atomic_json(out / 'controller.json', state)
        save()
        telemetry = Telemetry(out / 'telemetry.csv', 1)
        backend = JournalBackend(root, out, runtime, reference, state['backend_lifecycles'], save)
        telemetry.start()
        deadline = state['started_epoch'] + design['max_pipeline_seconds']
        try:
            profile = read(root / relative(reference['evaluation_profile_path']))
            write_json(out / 'model-verification.json',
                       verify_model(runtime['model'], root / relative(reference['model_manifest_path']), profile))
            for item in plan['schedule']:
                case = cases[item['task_id']]
                reserved = case['budget']['max_task_seconds'] + design['recovery_seconds'] + 120
                require(time.time() + reserved + (600 if backend.role is None else 0) <= deadline,
                        'Pipeline has no remaining time for the next complete attempt')
                socket = backend.ensure('base', deadline)
                state.update(stage='collecting', current_run=item)
                save()
                directory = out / 'private' / 'tasks' / relative(item['run_id'])
                trace = initialize(directory, case, item['arm'])
                advance(root, case['task'], case['protocol'], directory, trace, socket, sandbox, item['arm'],
                        journal_path=backend.journal_path, recovery_seconds=design['recovery_seconds'])
                backend.check_exclusive()
                outcome = finish(data, case['task'], case['source'], case['protocol'], directory, trace, sandbox)
                outcome.update(run_id=item['run_id'], arm=item['arm'], backend_load=backend.number,
                               timeout_call=trace.get('timeout_call'),
                               accounting_drain_seconds=trace.get('accounting_drain_seconds', 0.0),
                               complete_success_within_budget=trace['eligible_verified_trajectory'])
                write_json(directory / 'result.json', outcome)
                state['runs'].append(outcome)
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
                             frozen_files_unchanged=all(sha256(root / relative(n)) == h
                                                        for n, h in plan['frozen_files'].items()))
                save()
    return state
