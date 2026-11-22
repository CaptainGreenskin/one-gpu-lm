"""Collect expanded, disjoint training forks with one protected audit per block."""

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, single_writer, utc
from one_gpu_lm.verified.continuation_training_data import BASELINE_REVIEW, read
from one_gpu_lm.verified.decision_state_collection import DecisionCase, require
from one_gpu_lm.verified.failure_states import source_state
from one_gpu_lm.verified.first_edit_continuation import message_hash
from one_gpu_lm.verified.recovery_sft_projection import ORIGINAL_PROTOCOL
from one_gpu_lm.verified.recovery_state import inventory
from one_gpu_lm.verified.tasks import relative, validate_source


DIR = Path(__file__).resolve().parent
DESIGN = DIR / 'design.json'
PARENT = Path('experiments/044-deployment-recovery-sft/design.json')
RUNTIME = Path('private/parent-runtime.json')
REPLAY_DIR = Path('private/preference-expansion-replay-001')
PRIOR_PLAN = Path('private/first-action-045-retry-002/protocol.json')
PRIOR_AUDIT = Path('runs/q045-first-action-forks/audit.json')
OUTPUT = Path('runs/q047-expanded-forks')
ENGINE = DIR.parent / '045-first-action-forks'
SNAPSHOT_FILES = {'before.json', 'manifest.json', 'prefix.json'}


def load_engine():
    spec = importlib.util.spec_from_file_location('first_action_engine', ENGINE / 'run.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ENGINE))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def registered_blocks(sizes):
    first = 1
    for block_number, size in enumerate(sizes, 1):
        name = f'block-{block_number:02d}'
        yield f'blocks/{name}.json', str(OUTPUT / name), list(range(first, first + size))
        first += size


def case_from_snapshot(root, row, task, source, shared, total_budget):
    snapshot = root / REPLAY_DIR / 'restored' / f"t{row['ordinal']:03d}"
    hashes = row['snapshot_files_sha256']
    require(not snapshot.is_symlink() and set(hashes) == SNAPSHOT_FILES
            and all(sha256(snapshot / filename) == checksum for filename, checksum in hashes.items())
            and task['split'] == source['split'] == 'training'
            and task['family'] == source['family'] == row['family']
            and source['approved_for']['training'] is True
            and source['notice_review_complete'] is source['lineage_review_complete'] is True,
            'Restored state or approved training scope differs')
    prefix = read(snapshot / 'prefix.json')
    manifest = read(snapshot / 'manifest.json')
    before = read(snapshot / 'before.json')
    messages = prefix['messages']
    cutoff = prefix['historical_calls']
    charged = prefix['historical_generated_tokens']
    expected_budget = dict(total_budget, max_model_calls=total_budget['max_model_calls'] - cutoff,
                           max_total_generated_tokens=total_budget['max_total_generated_tokens'] - charged)
    source_sha = prefix['source_sha256']
    require(cutoff == row['opening_calls'] and cutoff in (1, 2) and len(messages) == 2 + 2 * cutoff
            and message_hash(messages) == prefix['messages_sha256']
            and prefix['prompt_tokens'] == row['prompt_tokens']
            and 0 < row['prompt_tokens'] <= total_budget['max_input_tokens']
            and 0 < charged <= cutoff * total_budget['max_generated_tokens_per_call']
            and prefix['budget'] == expected_budget
            and min(expected_budget['max_model_calls'], expected_budget['max_total_generated_tokens']) > 0
            and before == {name: item['sha256'] for name, item in manifest.items() if item['kind'] == 'file'}
            and source_sha == {name: before[name] for name in task['editable_files']}
            and inventory(snapshot / 'workspace') == manifest
            and source_state(snapshot / 'workspace', task) == source_sha
            and not set(task['hidden_checks']).intersection(manifest), 'Restored inputs or consumed budgets differ')
    proof = {'prefix_sha256': hashes['prefix.json'], 'request_messages_sha256': prefix['messages_sha256'],
             'packed_messages_sha256': prefix['packed_messages_sha256'], 'arms': {'control': prefix}}
    state = {'kind': 'early_edit', 'verified_source': None, 'decision_source_sha256': source_sha,
             'original_source_sha256': source_sha}
    return DecisionCase(task, source, snapshot, manifest, state, proof, messages,
                        dict(shared, evaluation_budget=expected_budget))


def cases_from_plan(root, plan, design):
    data = Path(plan['data'])
    task_manifest = read(data / 'tasks.json')
    source_manifest = read(data / 'sources.json')
    tasks = {task['id']: task for task in task_manifest['tasks']}
    sources = {source['repository']: source for source in source_manifest['sources']}
    require(len(tasks) == len(task_manifest['tasks']) and len(sources) == len(source_manifest['sources']),
            'Task or source identity is ambiguous')
    parent = read(root / PARENT)
    shared = {'model_id': design['model_id'], 'revision': design['base_revision'],
              'evaluation_profile_sha256': design['evaluation_profile_sha256'],
              'agent_policy': parent['agent_policy']}
    cases = {}
    for row in plan['rows']:
        task = tasks[row['task_id']]
        cases[task['id']] = case_from_snapshot(root, row, task, sources[task['repository']], shared,
                                               parent['actor_budget'])
    return data, cases


def freeze(args):
    from transformers import AutoTokenizer

    root = args.root
    engine = load_engine()
    from prepare import prompt_tokens

    design, parent = read(DESIGN), read(root / PARENT)
    replay_path = root / REPLAY_DIR / 'readiness.json'
    replay, replay_state = read(replay_path), read(replay_path.with_name('state.json'))
    prior, prior_audit = read(root / PRIOR_PLAN), read(root / PRIOR_AUDIT)
    rows = replay['replayed']
    require(sha256(root / PARENT) == design['parent_training_design_sha256']
            and sha256(root / PRIOR_PLAN) == design['prior_protocol_sha256']
            and sha256(root / PRIOR_AUDIT) == design['prior_audit_sha256']
            and sha256(replay_path) == design['replay_sha256'] == replay_state['replay_sha256']
            and replay_state['phase'] == 'complete' and replay_state['replayed_states'] == len(rows)
            and replay['protected_tool_replay'] is True and not replay['rejected']
            and prior_audit['passed'] is prior['training_only'] is prior_audit['training_only'] is True
            and prior_audit['protocol_sha256'] == design['prior_protocol_sha256']
            and not any(doc['final_test_opened'] for doc in (design, parent, prior, prior_audit, replay, replay_state))
            and replay['model_generations'] == replay['optimizer_updates'] == prior_audit['optimizer_updates'] == 0
            and engine.frozen(root, prior), 'The original verified training-only replay or runtime changed')
    old = {row['task_id'] for row in prior['rows']}
    ids = {row['task_id'] for row in rows}
    families = {row['family'] for row in rows}
    require(len(ids) == len(rows) == design['expected_training_states'] == sum(design['block_sizes'])
            and len(families) == design['expected_training_families'] and not old.intersection(ids)
            and [row['ordinal'] for row in rows] == list(range(1, len(rows) + 1)),
            'The frozen cohort must contain every disjoint replay in the original order')

    original_path = root / ORIGINAL_PROTOCOL
    historical = read(original_path)
    historical_root = root / relative(historical['run_directory'])
    audit_path = historical_root / 'audit.json'
    require(sha256(original_path) == parent['collection']['historical_protocol_sha256']
            and sha256(audit_path) == parent['collection']['historical_audit_sha256'],
            'The original training trajectories have a different audit identity')
    historical_rows = {row['id']: row for row in read(audit_path)['rows']}
    collection = Path(prior['collection'])
    data_location = relative(historical['data_directory'])
    data = collection / data_location
    review = read(root / BASELINE_REVIEW)
    task_hash, source_hash = sha256(data / 'tasks.json'), sha256(data / 'sources.json')
    require(task_hash == review['task_manifest_sha256'] == historical['frozen_files'][str(data_location / 'tasks.json')]
            and source_hash == review['source_manifest_sha256'] == historical['frozen_files'][str(data_location / 'sources.json')],
            'Source approval does not cover the exact task and source manifests')
    files = {**prior['frozen_files'], str(PRIOR_PLAN): sha256(root / PRIOR_PLAN),
             str(PRIOR_AUDIT): sha256(root / PRIOR_AUDIT), str(replay_path.relative_to(root)): sha256(replay_path),
             str(replay_path.with_name('state.json').relative_to(root)): sha256(replay_path.with_name('state.json'))}
    for path in (DESIGN, Path(__file__), ENGINE / 'run.py', ENGINE / 'prepare.py'):
        files[str(path.relative_to(root))] = sha256(path)
    external = dict(prior['runtime_inputs'])
    external.update({str(data / 'tasks.json'): task_hash, str(data / 'sources.json'): source_hash,
                     str(audit_path): sha256(audit_path)})
    common = {'schema_version': 1, 'training_only': True, 'final_test_opened': False,
              'collection': str(collection), 'data': str(data), 'design_sha256': sha256(DESIGN),
              'deployment_inputs': prior['deployment_inputs'], 'runtime_inputs': external, 'frozen_files': files}
    _, cases = cases_from_plan(root, dict(common, rows=rows), design)
    profile_path = Path(common['deployment_inputs']['evaluation_profile'])
    profile = read(profile_path)
    require(sha256(profile_path) == design['evaluation_profile_sha256']
            and (profile['model_id'], profile['revision']) == (design['model_id'], design['base_revision'])
            and not profile.get('generation_modes') and profile['generation']['do_sample'] is False,
            'Both arms require the pinned greedy suffix profile')
    tokenizer = AutoTokenizer.from_pretrained(read(root / RUNTIME)['model'], local_files_only=True, trust_remote_code=False)
    reviewed = set()
    for case in cases.values():
        task, source = case.task, case.source
        repository = task['repository']
        if repository not in reviewed:
            require(repository in review['sources'], 'Training repository approval missing')
            validate_source(source, data / 'snapshots' / relative(repository.replace('/', '--')), 'training')
            reviewed.add(repository)
        evidence = historical_rows[task['id']]
        prefix = case.proof['arms']['control']
        trace = historical_root / 'private/tasks' / relative(task['id']) / 'trajectory.json'
        require(evidence['complete'] and evidence['input_replay_passed']
                and evidence['trajectory_sha256'] == prefix['historical_trajectory_sha256'] == sha256(trace)
                and read(trace)['split'] == 'training'
                and prompt_tokens(case.messages, tokenizer, profile, parent['agent_policy'], case.budget)
                == {key: prefix[key] for key in ('prompt_tokens', 'prompt_ids_sha256', 'packed_messages_sha256')},
                'Historical actor or actual tokenizer prefix differs')

    with single_writer(args.plan.with_suffix('.lock')):
        destination = args.plan.parent / 'blocks'
        require(not args.plan.exists() and not destination.exists() and not (root / OUTPUT).exists(),
                'Preserve the original block registration and run')
        with TemporaryDirectory(prefix='.blocks-', dir=args.plan.parent) as staging:
            temp = Path(staging) / 'blocks'
            temp.mkdir()
            blocks = {}
            for name, output, ordinals in registered_blocks(design['block_sizes']):
                block = temp / Path(name).name
                atomic_json(block, dict(common, output=output, rows=[rows[n - 1] for n in ordinals]))
                blocks[name] = sha256(block)
            temp.rename(destination)
        atomic_json(args.plan, {'schema_version': 1, 'design_sha256': sha256(DESIGN), 'blocks': blocks,
                                'training_only': True, 'optimizer_updates': 0, 'final_test_opened': False})
    return {'protocol_sha256': sha256(args.plan), 'training_states': len(rows), 'training_families': len(families),
            'blocks': design['block_sizes'], 'physical_rollouts': 2 * len(rows),
            'historical_calls': dict(Counter(row['opening_calls'] for row in rows)),
            'training_calls_ceiling': 2 * sum(case.budget['max_model_calls'] for case in cases.values()),
            'generated_tokens_ceiling': 2 * sum(case.budget['max_total_generated_tokens'] for case in cases.values()),
            'model_generations': 0, 'optimizer_updates': 0, 'final_test_opened': False}


def verified_campaign(path, publication):
    plan, design, receipt = read(path), read(DESIGN), read(publication)
    expected = list(registered_blocks(design['block_sizes']))
    require(receipt['protocol_sha256'] == sha256(path) and receipt['anonymous_files_verified'] is True
            and receipt['ci_conclusion'] == 'success' and plan['design_sha256'] == sha256(DESIGN)
            and plan['training_only'] is True and plan['final_test_opened'] is False
            and plan['optimizer_updates'] == 0 and set(plan['blocks']) == {name for name, _, _ in expected},
            'Require the verified public experiment and its complete schedule')
    return plan, design, expected


def load_block(root, path, publication, engine):
    campaign, design, expected = verified_campaign(path.parent.parent / 'protocol.json', publication)
    key = str(path.relative_to(path.parent.parent))
    choices = {name: (output, ordinals) for name, output, ordinals in expected}
    require(key in choices and sha256(path) == campaign['blocks'][key], 'Block is outside the frozen campaign')
    output, ordinals = choices[key]
    plan = read(path)
    require(plan['training_only'] is True and plan['final_test_opened'] is False
            and plan['design_sha256'] == sha256(DESIGN) and plan['output'] == output
            and [row['ordinal'] for row in plan['rows']] == ordinals and engine.frozen(root, plan),
            'Block order or immutable inputs differ')
    data, cases = cases_from_plan(root, plan, design)
    require(len(cases) == len(ordinals), 'Block contains a duplicate task')
    return plan, design, data, cases


def audited_block(directory, protocol_sha, ordinals):
    target = directory / 'audit.json'
    audit, control = read(target), read(directory / 'controller.json')
    records = directory / 'private/preference-pairs.json'
    require(audit['passed'] is audit['training_only'] is control['backend_stopped'] is True
            and audit['final_test_opened'] is audit['weights_public'] is False and audit['optimizer_updates'] == 0
            and audit['protocol_sha256'] == control['protocol_sha256'] == protocol_sha
            and audit['controller_sha256'] == sha256(directory / 'controller.json')
            and audit['audited_attempts'] == audit['expected_attempts'] == len(control['runs']) == 2 * len(ordinals)
            and [row['pair'] for row in audit['pairs']] == ordinals
            and audit['training_export_sha256'] == (sha256(records) if records.is_file() else None),
            'Physical block evidence does not match its complete assigned schedule')
    return {'pairs': len(ordinals), 'attempts': audit['audited_attempts'],
            'requests': audit['journal_requests_verified'], 'audit_sha256': sha256(target),
            'preference_pairs': sum(audit['eligible_contrasts'].values()),
            'successful_repairs': {arm: sum(row['arms'][arm]['verified_and_submitted'] for row in audit['pairs'])
                                   for arm in ('greedy', 'sampled')}}


def campaign(args):
    plan, design, blocks = verified_campaign(args.plan, args.publication)
    output = args.root / OUTPUT
    state_path = output / 'campaign.json'
    with single_writer(output / 'campaign.lock'):
        original = read(state_path) if state_path.is_file() else {}
        require(not (output / 'audit.json').exists() and original.get('protocol_sha256', sha256(args.plan)) == sha256(args.plan),
                'Preserve the terminal campaign and its protocol')
        state = {'protocol_sha256': sha256(args.plan), 'phase': 'running',
                 'started_utc': original.get('started_utc', utc()), 'blocks': {},
                 'expected_attempts': 2 * design['expected_training_states'], 'optimizer_updates': 0,
                 'final_test_opened': False}
        interpreter = [sys.executable, '-u', str(DIR / 'run.py')]
        try:
            for name, where, ordinals in blocks:
                block = args.plan.parent / relative(name)
                directory = args.root / relative(where)
                state['active_block'] = directory.name
                atomic_json(state_path, state)
                flags = ['--root', str(args.root), '--plan', str(block), '--publication', str(args.publication)]
                if not directory.exists():
                    subprocess.run([*interpreter, 'run', *flags], check=True)
                if not (directory / 'audit.json').exists():
                    control = read(directory / 'controller.json')
                    require(control['complete'] and control['backend_stopped'], 'An interrupted block requires diagnosis')
                    subprocess.run([*interpreter, 'audit', *flags], check=True,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
                state['blocks'][directory.name] = audited_block(directory, plan['blocks'][name], ordinals)
            state.update(phase='audited', finished_utc=utc(), active_block=None)
            metrics = tuple(state['blocks'].values())
            terminal = dict(state, audited_attempts=sum(item['attempts'] for item in metrics),
                            verified_requests=sum(item['requests'] for item in metrics),
                            qualifying_preference_pairs=sum(item['preference_pairs'] for item in metrics),
                            weights_public=False)
            atomic_json(output / 'audit.json', terminal)
            return terminal
        except BaseException as error:
            state.update(phase='failed', error_type=type(error).__name__)
            raise
        finally:
            atomic_json(state_path, state)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'campaign', 'run', 'serve', 'audit'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--publication', type=Path)
    parser.add_argument('--socket', type=Path)
    args = parser.parse_args()
    args.root, args.plan = args.root.resolve(), args.plan.resolve()
    if args.mode == 'freeze':
        result = freeze(args)
    elif args.publication is None or (args.mode == 'serve' and args.socket is None):
        parser.error('The verified publication and any required socket are mandatory')
    elif args.mode == 'campaign':
        result = campaign(args)
    else:
        engine = load_engine()
        inputs = load_block(args.root, args.plan, args.publication, engine)
        if args.mode == 'run':
            result = engine.collect(args, inputs=inputs, entrypoint=DIR / 'run.py')
        elif args.mode == 'serve':
            result = engine.serve(args, inputs=inputs)
        else:
            result = engine.audit(args, inputs=inputs)
    if result is not None:
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
