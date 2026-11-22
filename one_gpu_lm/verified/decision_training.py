"""Register, train and audit one LoRA from the complete preference collection."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
from tempfile import TemporaryDirectory
import time

from one_gpu_lm.common.checkpoints import load_checkpoint
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, gpu_lock_path, identity_digest, single_writer, utc
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.adapter_bundle import FILES, restore_adapter, verify_adapter
from one_gpu_lm.models.hf_runtime import Runtime
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.training.decision_preference import action_log_probability, encode_decision
from one_gpu_lm.training.preference_epoch import PreferenceRecipe, replay_history, run_epoch, schedule
from one_gpu_lm.training.state import state_digest
from .candidate_handoff import TRAINING_RECEIPTS
from .continuation_training import unchanged
from .decision_training_data import review_inputs
from .expanded_preference import EXPANSION_PATH, extend_review
from .runtime_identity import verify as verify_environment
from .tasks import relative


KIND = 'audited_action_preference_lora'
PARENT = 'experiments/031-correction-mix/candidate-protocol.json'
RUNTIME = 'private/parent-runtime.json'
RECIPE_BETA = 0.1
PAIRS_PER_UPDATE = 4


def read(path):
    return json.loads(Path(path).read_bytes())


def base_tokenizer(root):
    from transformers import AutoTokenizer

    model = Path(read(root/RUNTIME)['model'])
    return model, AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)


def freeze(root, protocol, args):
    if protocol.exists() or not protocol.is_relative_to(root/'private'):
        raise ValueError('Use a new private preference registration')
    destination = relative(args.run_directory.as_posix())
    if destination.parts[0] != 'runs' or (root/destination).exists():
        raise ValueError('Use an unused preference training run directory')
    _, tokenizer = base_tokenizer(root)
    identifiers = read(args.private_identifiers) if getattr(args,'private_identifiers',None) else ()
    if not isinstance(identifiers,(list,tuple)) or any(not isinstance(value,str) for value in identifiers):
        raise ValueError('Private identifier checks require an array of strings')
    selected = review_inputs(args.collection_root, args.collection_protocol, args.collection_protocol_sha256,
                             args.collection_audit_sha256, args.development_cohort, args.development_cohort_sha256,
                             tokenizer,private_identifiers=identifiers)
    expansion = getattr(args, 'expanded_collection_protocol', None)
    results_publication = getattr(args, 'expanded_results_publication', None)
    if (expansion is None) != (results_publication is None):
        raise ValueError('Supply both the complete expanded collection and its verified result publication')
    if expansion is not None:
        selected = extend_review(selected, args.collection_root, args.collection_protocol, expansion,
                                 results_publication, args.development_cohort, tokenizer, private_identifiers=identifiers)
    parent = read(root/PARENT)
    profile_path = root/relative(parent['profile_path'])
    profile = read(profile_path)
    settings = profile['training']
    inherited = read(root/relative(parent['cycle_spec']))['candidate']
    if selected.profile != profile or settings['epochs'] != 1 or not settings['gradient_checkpointing']:
        raise ValueError('The registered collection must use the pinned LoRA training profile')
    recipe = PreferenceRecipe(settings['seed'], RECIPE_BETA, PAIRS_PER_UPDATE,
                              settings['learning_rate'], settings['max_grad_norm'])
    batches = schedule(selected.choices, recipe)
    inputs = protocol.with_name(protocol.stem+'-inputs')
    protocol.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    names = {'data': 'pairs.json', 'sources': 'sources.json', 'review': 'review.json'}
    expected = {'data': selected.review['actual_training_data_sha256'],
                'sources': selected.review['source_manifest_sha256']}
    if not inputs.exists():
        with TemporaryDirectory(prefix='preference-inputs-', dir=inputs.parent) as area:
            staged = Path(area)/'selected'
            staged.mkdir(mode=0o700)
            for key, digest in expected.items():
                target = staged/names[key]
                if key == 'data' and selected.data_bytes is not None:
                    target.write_bytes(selected.data_bytes)
                else:
                    shutil.copyfile(selected.paths[key], target)
                if sha256(target) != digest:
                    raise ValueError('Copied preference input differs from the verified digest')
            atomic_json(staged/names['review'], selected.review)
            staged.rename(inputs)
    if (any(sha256(inputs/names[key]) != digest for key, digest in expected.items())
            or read(inputs/names['review']) != selected.review):
        raise ValueError('Preserve the original audited preference inputs')
    paths = {key: (inputs/name).relative_to(root).as_posix() for key, name in names.items()}
    evidence = [root/name for name in (PARENT, RUNTIME, parent['profile_path'], parent['environment_path'],
                                       parent['model_manifest_path'], parent['cycle_spec'], *paths.values())]
    evidence += list((root/'one_gpu_lm').rglob('*.py'))
    name = {EXPANSION_PATH.name:'048-expanded-action-preference',
            '045-first-action-forks':'046-first-action-preference'}.get(selected.review['collection_experiment'],
                                                                    '044-action-preference')
    plan = {'schema_version': 1, 'candidate': name, 'kind': KIND,
            'model_id': profile['model_id'], 'revision': profile['revision'],
            'profile_path': parent['profile_path'], 'sources_path': paths['sources'], 'data_path': paths['data'],
            'input_review_path': paths['review'], 'run_directory': str(destination),
            'environment_path': parent['environment_path'], 'model_manifest_path': parent['model_manifest_path'],
            'collection_protocol_sha256': args.collection_protocol_sha256,
            'collection_audit_sha256': args.collection_audit_sha256,
            'expected_trainable_parameters': parent['expected_trainable_parameters'],
            'attempt_limit_seconds': inherited['maximum_training_pipeline_seconds'],
            'recipe': asdict(recipe), 'planned_pairs': len(selected.choices), 'expected_optimizer_updates': len(batches),
            'order_sha256': identity_digest(batches), 'final_test_opened': False, 'weights_public': False,
            'frozen_files': {path.relative_to(root).as_posix(): sha256(path) for path in sorted(set(evidence))}}
    if 'expansion' in selected.review:
        plan['source_expansion'] = selected.review['expansion']
    with protocol.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return {'protocol_sha256': sha256(protocol), 'pairs': plan['planned_pairs'],
            'reverified_first_actions': selected.review['physical_first_actions_reverified'],
            'optimizer_updates': plan['expected_optimizer_updates'], 'final_test_opened': False}


def registered(root, protocol, publication):
    plan, receipt = read(protocol), read(publication)
    checksum = sha256(protocol)
    if (plan['kind'] != KIND or plan['final_test_opened'] is not False or plan['weights_public'] is not False
            or receipt['protocol_sha256'] != checksum or receipt['anonymous_files_verified'] is not True
            or receipt['ci_conclusion'] != 'success' or not re.fullmatch(r'[a-f0-9]{40}', receipt['commit'])
            or not unchanged(root, plan['frozen_files'])):
        raise ValueError('Expected the exact public executable registration and reviewed private inputs')
    keys = ('data_path', 'sources_path', 'input_review_path', 'profile_path', 'environment_path', 'model_manifest_path')
    if any(plan[key] not in plan['frozen_files'] for key in keys):
        raise ValueError('A training input is missing from the frozen inventory')
    data_path, source_path, review_path, profile_path, environment, manifest = (
        root/relative(plan[key]) for key in keys)
    review, profile = read(review_path), read(profile_path)
    if (review['collection_protocol_sha256'] != plan['collection_protocol_sha256']
            or review['collection_audit_sha256'] != plan['collection_audit_sha256']
            or review['actual_training_data_sha256'] != sha256(data_path)
            or review['source_manifest_sha256'] != sha256(source_path)
            or review.get('expansion') != plan.get('source_expansion')
            or review['final_test_opened'] is not False or review['weights_public'] is not False
            or review['known_pattern_findings'] or (profile['model_id'], profile['revision']) !=
            (plan['model_id'], plan['revision'])):
        raise ValueError('Preference input provenance changed')
    model, tokenizer = base_tokenizer(root)
    rows = read(data_path)
    choices = [encode_decision(row, tokenizer, profile) for row in rows]
    recipe = PreferenceRecipe(**plan['recipe'])
    batches = schedule(choices, recipe)
    if (len(choices) != plan['planned_pairs'] or len(batches) != plan['expected_optimizer_updates']
            or identity_digest(batches) != plan['order_sha256']):
        raise ValueError('The actual tokenizer or preference schedule differs from the registration')
    return plan, checksum, model, profile, environment, manifest, rows, choices, recipe, batches, review


def train(root, protocol, publication, *, resume=False, stop_requested=None):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    plan, checksum, base, profile, environment, manifest, rows, choices, recipe, _, _ = registered(root, protocol, publication)
    out = root/relative(plan['run_directory'])
    with single_writer(gpu_lock_path()):
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                       text=True, timeout=10).strip()
        if busy or out.exists() != resume:
            raise RuntimeError('Require the idle GPU and the matching new or resumable preference run')
        out.mkdir(mode=0o700, parents=True, exist_ok=resume)
        result_path = out/'result.json'
        if result_path.exists() and read(result_path)['complete']:
            raise ValueError('The completed preference candidate is immutable')
        attempt = len(list((out/'private').glob('attempt-*'))) + 1
        private = out/'private'/f'attempt-{attempt:04d}'
        private.mkdir(mode=0o700, parents=True)
        started = time.monotonic()

        def stop():
            return (time.monotonic() - started >= plan['attempt_limit_seconds']
                    or stop_requested is not None and stop_requested())

        result = {'schema_version': 1, 'stage': 'loading', 'attempt': attempt, 'started_utc': utc(), 'complete': False,
                  'protocol_sha256': checksum, 'profile_sha256': plan['frozen_files'][plan['profile_path']],
                  'data_sha256': plan['frozen_files'][plan['data_path']], 'development_scores': 0,
                  'final_test_opened': False, 'weights_public': False}
        atomic_json(result_path, result)
        telemetry = Telemetry(private/'telemetry.csv', attempt)
        telemetry.start()
        try:
            atomic_json(out/'environment.json', verify_environment(read(environment)))
            atomic_json(private/'verified-model.json', verify_model(base, manifest, profile))
            runtime = Runtime(base, profile, private)
            torch.manual_seed(recipe.seed)
            random.seed(recipe.seed)
            model = get_peft_model(runtime.model, LoraConfig(task_type=TaskType.CAUSAL_LM, **profile['lora']))
            model.peft_config['default'].base_model_name_or_path = profile['model_id']
            model.peft_config['default'].revision = profile['revision']
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            result.update(stage='training')
            atomic_json(result_path, result)
            summary = run_epoch(model, choices, recipe, out, {'protocol_sha256': checksum},
                                device='cuda:0', resume=resume, interrupted=stop,
                                expected_trainable_parameters=plan['expected_trainable_parameters'])
            result.update(summary, optimizer_steps=summary['optimizer_updates'])
            if summary['complete']:
                identity = read(out/'preference-identity.json')
                checkpoint, trainer, checkpoint_manifest = load_checkpoint(out, identity)
                hashes = {name: checkpoint_manifest['files']['adapter/'+name] for name in FILES}
                adapter = out/'private/adapter'
                if not adapter.exists():
                    restore_adapter(checkpoint/'adapter', adapter, profile, hashes)
                verified = verify_adapter(adapter, profile, hashes)
                if verified['parameters'] != summary['trainable_parameters'] or set(p.name for p in adapter.iterdir()) != set(FILES):
                    raise ValueError('Checkpoint and exported LoRA tensor inventory differ')
                atomic_json(out/'adapter.json', {'files': hashes, 'weights_public': False, 'final_test_opened': False})
                history = trainer['history']
                atomic_json(out/'updates.json', [{k: v for k, v in step.items() if k != 'records'}
                                                  | {'pairs': len(step['records'])} for step in history])
                model.gradient_checkpointing_disable()
                runtime.model = model.eval()
                probe = {'messages': rows[0]['messages'], 'max_new_tokens': 16,
                         'agent_policy': {'format': 'native_roundtrip', 'context': 'bounded_history_v1'}}

                def action_scores():
                    choice = choices[0]
                    with torch.no_grad():
                        return [float(action_log_probability(runtime.model, ids, choice.prefix_tokens, 'cuda:0'))
                                for ids in (choice.chosen, choice.rejected)]

                scores = action_scores()
                before = runtime.generate(probe)['ids']
                runtime.model = PeftModel.from_pretrained(model.unload(), adapter, is_trainable=False).eval()
                after = runtime.generate(probe)['ids']
                same_scores = all(math.isclose(a, b, rel_tol=1e-5, abs_tol=2e-3)
                                  for a, b in zip(scores, action_scores()))
                if not before or before != after or not same_scores:
                    raise ValueError('Reloaded preference checkpoint changed the training-only probe')
                result.update(stage='completed', all_selected_records_consumed=True,
                              saved_reload_greedy_ids_equal=True, saved_reload_action_scores_equal=True,
                              reload_probe_ids_sha256=identity_digest(after))
            else:
                reason = 'attempt_time_budget' if time.monotonic() - started >= plan['attempt_limit_seconds'] else 'signal'
                result.update(stage='paused', reason=reason)
        except BaseException as error:
            result.update(stage='failed', complete=False, error_type=type(error).__name__)
            raise
        finally:
            result.update(telemetry=telemetry.stop(), finished_utc=utc())
            result['frozen_files_unchanged'] = unchanged(root, plan['frozen_files'])
            if not result['frozen_files_unchanged']:
                result.update(stage='inputs_changed', complete=False)
            atomic_json(result_path, result)
    return result


def audit(root, protocol, publication):
    import torch
    from safetensors.torch import load_file

    plan, checksum, _, profile, environment, _, _, _, _, batches, review = registered(root, protocol, publication)
    out = root/relative(plan['run_directory'])
    result = read(out/'result.json')
    identity = read(out/'preference-identity.json')
    if (result['stage'] != 'completed' or result['complete'] is not True or result['final_test_opened']
            or result['weights_public'] or result['development_scores']
            or result['frozen_files_unchanged'] is not True or result['saved_reload_greedy_ids_equal'] is not True
            or result['saved_reload_action_scores_equal'] is not True
            or result['all_selected_records_consumed'] is not True or result['protocol_sha256'] != checksum
            or result['data_sha256'] != plan['frozen_files'][plan['data_path']]
            or result['profile_sha256'] != plan['frozen_files'][plan['profile_path']]
            or result['optimizer_steps'] != len(batches) or result['pairs_consumed'] != plan['planned_pairs']
            or identity['inputs'] != {'protocol_sha256': checksum} or identity['device_type'] != 'cuda'
            or identity['ordered_pairs_sha256'] != plan['order_sha256'] or identity['recipe'] != plan['recipe']
            or read(out/'environment.json') != read(environment)):
        raise ValueError('Require one completed registered training-only preference run')
    checkpoint, trainer, manifest = load_checkpoint(out, identity)
    history = trainer['history']
    replay_history(batches, history)
    updates = [{k: v for k, v in step.items() if k != 'records'} | {'pairs': len(step['records'])} for step in history]
    selection = {'pairs': plan['planned_pairs'], 'updates': len(batches), 'order_sha256': plan['order_sha256'],
                 'batches': batches, 'final_test_opened': False}
    if (trainer['step'] != len(history) or len(history) != len(batches) or read(out/'curve.json') != history
            or read(out/'updates.json') != updates or read(out/'selection.json') != selection
            or trainer['initial_adapter_sha256'] == trainer['adapter_state_sha256']):
        raise ValueError('Preference optimizer history does not cover the exact registered schedule')
    state = torch.load(checkpoint/'state.pt', map_location='cpu', weights_only=True)
    adapter_state = load_file(checkpoint/'adapter/adapter_model.safetensors', device='cpu')
    for name, contents in [('adapter', adapter_state), ('optimizer', state['optimizer']), ('rng', state['rng'])]:
        key = name+'_state_sha256'
        if state_digest(contents) != trainer[key] or trainer[key] != result[key]:
            raise ValueError('Committed preference checkpoint state differs from the training receipt')
    hashes = {name: manifest['files']['adapter/'+name] for name in FILES}
    if read(out/'adapter.json') != {'files': hashes, 'weights_public': False, 'final_test_opened': False}:
        raise ValueError('Preference adapter receipt differs from its committed weights')
    verified = verify_adapter(out/'private/adapter', profile, hashes)
    if verified['parameters'] != plan['expected_trainable_parameters']:
        raise ValueError('Preference adapter changed the registered LoRA capacity')
    target = out/'integrity.json'
    audited_utc = read(target)['audited_utc'] if target.exists() else utc()
    evidence = {'schema_version': 1, 'passed': True, 'protocol_sha256': checksum, 'audited_utc': audited_utc,
                'adapter_files_sha256': hashes, 'optimizer_updates': len(history), 'adapter_cpu_verification': verified,
                'receipt_sha256': {name: sha256(out/name) for name in sorted(TRAINING_RECEIPTS)},
                'audit_model_generations': 0, 'audit_optimizer_updates': 0, 'final_test_opened': False,
                'weights_public': False}
    if target.exists() and read(target) != evidence:
        raise ValueError('Preserve the original preference candidate integrity receipt')
    if not target.exists():
        atomic_json(target, evidence)
    candidate = {**review, 'experiment': plan['candidate'], 'checked_utc': audited_utc,
                 'model_id': profile['model_id'], 'revision': profile['revision'],
                 'audited_checkpoint_matches': True, 'eligible_for_existing_scope_of_development': True,
                 'training_integrity_sha256': sha256(target), 'checkpoint_files_sha256': hashes}
    review_path = out/'candidate-input-review.json'
    if review_path.exists() and read(review_path) != candidate:
        raise ValueError('Preserve the original preference candidate review')
    if not review_path.exists():
        atomic_json(review_path, candidate)
    return evidence


def main():
    os.umask(0o077)
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, required=True)
    cli.add_argument('--protocol', type=Path, required=True)
    modes = cli.add_subparsers(dest='mode', required=True)
    preparation = modes.add_parser('freeze')
    for name in ('collection-root', 'collection-protocol', 'development-cohort', 'run-directory'):
        preparation.add_argument('--'+name, type=Path, required=True)
    for name in ('collection-protocol-sha256', 'collection-audit-sha256', 'development-cohort-sha256'):
        preparation.add_argument('--'+name, required=True)
    preparation.add_argument('--private-identifiers',type=Path)
    preparation.add_argument('--expanded-collection-protocol',type=Path)
    preparation.add_argument('--expanded-results-publication',type=Path)
    for mode in ('train', 'audit'):
        command = modes.add_parser(mode)
        command.add_argument('--publication', type=Path, required=True)
        if mode == 'train':
            command.add_argument('--resume', action='store_true')
    args = cli.parse_args()
    root, protocol = args.root.resolve(), args.protocol.resolve()
    if args.mode == 'freeze':
        result = freeze(root, protocol, args)
    elif args.mode == 'train':
        signals = []
        for kind in (signal.SIGINT, signal.SIGTERM):
            signal.signal(kind, lambda number, frame: signals.append(number))
        result = train(root, protocol, args.publication, resume=args.resume, stop_requested=lambda: bool(signals))
        print(json.dumps({key: result[key] for key in ('stage', 'optimizer_steps', 'final_test_opened')}))
        raise SystemExit(0 if result['complete'] else 75)
    else:
        value = audit(root, protocol, args.publication)
        result = {key: value[key] for key in ('passed', 'optimizer_updates', 'final_test_opened')}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
