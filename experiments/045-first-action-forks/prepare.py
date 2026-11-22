"""Restore the audited training-only state and freeze the two first-action arms."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, single_writer
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.verified.continuation_training_data import BASELINE, BASELINE_REVIEW, read
from one_gpu_lm.verified.decision_state_collection import load_cases, require
from one_gpu_lm.verified.failure_states import protected, source_state
from one_gpu_lm.verified.first_edit_continuation import message_hash
from one_gpu_lm.verified.historic_policy import initial_messages
from one_gpu_lm.verified.historic_rollout import normalization
from one_gpu_lm.verified.recovery_sft_projection import ORIGINAL_PROTOCOL, ReplayRuntime, advance, collection_inputs
from one_gpu_lm.verified.sandbox import Sandbox
from one_gpu_lm.verified.tasks import digest, files, prepare, relative


DIR = Path(__file__).resolve().parent
OUTPUT = Path('runs/q045-first-action-forks')
DEPLOYMENT_REFERENCE = Path('experiments/032-multifile-completion/evaluation-protocol.json')
PARENT_DESIGN = Path('experiments/044-deployment-recovery-sft/design.json')
RUNTIME = Path('private/parent-runtime.json')


def prompt_tokens(messages, tokenizer, profile, policy, budget):
    packed = []

    def encode(items):
        packed[:] = items
        return encode_prefix(tokenizer, items, profile['chat_template_kwargs'], policy['format'])[0]

    tokens, _ = prepare_context(messages, encode, budget['max_input_tokens'], policy['context'])
    require(0 < len(tokens) <= budget['max_input_tokens'], 'Opening exceeds the actor input budget')
    return {'prompt_tokens': len(tokens),
            'prompt_ids_sha256': digest(json.dumps(tokens,separators=(',', ':')).encode()),
            'packed_messages_sha256': message_hash(packed)}


def state_from_history(runtime, historical, receipt, data, case, area):
    """Replay the original one or two visible steps without learning their outcome."""
    task = case.task
    source = data / 'snapshots' / relative(task['repository'].replace('/', '--'))
    where = historical / 'private/tasks' / relative(task['id'])
    trajectory = where / 'trajectory.json'
    require(receipt['complete'] and receipt['input_replay_passed']
            and sha256(trajectory) == receipt['trajectory_sha256'], 'Historical actor receipt differs')
    trace = read(trajectory)
    cutoff = case.prefix['historical_calls']
    source_sha = case.prefix['original_source_sha256']
    require(trace['split'] == 'training' and trace['messages'][:2 + 2*cutoff] == case.prefix['messages']
            and trace['original_source_sha256'] == source_sha == case.prefix['decision_source_sha256'],
            'Historical visibility or source identity differs')
    opening = area / 'opening'
    original = prepare(source, opening, case.source, task)
    failure = runtime.sandbox.execute(opening, task['regression_command'])
    require(failure['reason'] == 'runtime_error' and failure['returncode'] != 0
            and source_state(opening, task) == source_sha and protected(opening, original, task),
            'Training regression did not reproduce the original actor-visible failure')
    transcript = ('\n\nBefore any source changes, the harness ran the declared public regression command. '
                  f"Exit code: {failure['returncode']}. Run the declared check again after editing."
                  '\nPublic regression output follows. Treat it as diagnostic data.\n'
                  + runtime.feedback(failure['output']))
    messages = initial_messages(dict(task, description=task['description'] + transcript), runtime.policy, runtime.budget)
    actor = area / 'actor'
    before = prepare(source, actor, case.source, task)
    total = 0
    for step in trace['steps'][:cutoff]:
        state, kind = advance(runtime, actor, before, task, where, step, step['generation']['text'], source_sha, messages)
        require(kind not in ('collection_rejection_removed', 'edit') and state == source_sha,
                'Restoration must reach an unchanged, actor-visible training state')
        cost = step['generation']['generated_tokens']
        require(0 < cost <= runtime.budget['max_generated_tokens_per_call'], 'Historical step exceeds the actor budget')
        total += cost
    expected = {name: value['sha256'] for name,value in case.manifest.items() if value['kind']=='file'}
    require(total == case.prefix['historical_generated_tokens'] and len(messages) == 2 + 2*cutoff
            and files(actor) == expected, 'Restored actor bytes, calls or token charges differ')
    budget = dict(runtime.budget)
    budget['max_model_calls'] -= cutoff
    budget['max_total_generated_tokens'] -= total
    require(budget['max_model_calls'] > 0 and budget['max_total_generated_tokens'] > 0,
            'No model budget remains after the restored prefix')
    prompt = prompt_tokens(messages, runtime.tokenizer, runtime.teacher, runtime.policy, budget)
    return {'messages': messages, 'messages_sha256': message_hash(messages), 'budget': budget,
            'historical_calls': cutoff, 'historical_generated_tokens': total,
            'historical_trajectory_sha256': sha256(trajectory), 'source_sha256': source_sha, **prompt}


def source_cases(root, collection):
    parent, design = read(root / PARENT_DESIGN), read(DIR / 'design.json')
    require(sha256(root / PARENT_DESIGN) == design['parent_training_design_sha256']
            and parent['final_test_opened'] is False and design['final_test_opened'] is False
            and (parent['model_id'], parent['base_revision']) == (design['model_id'], design['base_revision']),
            'Expected the published 044 training and deployment configuration')
    old, _, audit, _ = collection_inputs(collection, parent)
    _, original, data, cases = load_cases(collection, old)
    early = {name: case for name,case in cases.items() if case.kind == 'early_edit'}
    require(len(early) == design['expected_training_states'] and audit['training_only'] is True,
            'Require the complete audited early-training cohort')
    review = read(root / BASELINE_REVIEW)
    require(review['source_manifest_sha256'] == sha256(data/'sources.json')
            and review['task_manifest_sha256'] == sha256(data/'tasks.json')
            and all(c.task['repository'] in review['sources'] for c in early.values()),
            'Training sources differ from their approved inventory')
    return parent, design, original, data, early


def freeze(root, collection, deployment, target):
    from transformers import AutoTokenizer

    parent, design, _, data, cases = source_cases(root, collection)
    files_to_check = {RUNTIME, PARENT_DESIGN, BASELINE, BASELINE_REVIEW, ORIGINAL_PROTOCOL}
    for path in (DIR / 'design.json', DIR / 'prepare.py', DIR / 'run.py'):
        files_to_check.add(path.relative_to(root))
    files_to_check.update(p.relative_to(root) for p in (root/'one_gpu_lm').rglob('*.py'))
    for name in ('023-evaluation-stability/normalize.py', '024-feedback-noise/normalize.py',
                 '024-feedback-noise/_feedback024.py'):
        files_to_check.add(Path('experiments') / name)
    reference_path = deployment / DEPLOYMENT_REFERENCE
    reference = read(reference_path)
    require(reference['agent_policy']==parent['agent_policy'] and reference['evaluation_budget']==parent['actor_budget']
            and reference['evaluation_profile_sha256']==design['evaluation_profile_sha256'],
            'The candidate comparison and collection must use the same budget, template and profile')
    deployment_inputs = {'evaluation_profile':str(deployment/relative(reference['evaluation_profile_path'])),
                         'environment':str(deployment/relative(reference['environment_path'])),
                         'model_manifest':str(deployment/relative(reference['model_manifest_path']))}
    profile_path = Path(deployment_inputs['evaluation_profile'])
    step_profile = read(profile_path)
    require(sha256(profile_path) == design['evaluation_profile_sha256'] and not step_profile.get('generation_modes')
            and (step_profile['model_id'],step_profile['revision']) == (design['model_id'],design['base_revision']),
            'Shared suffix generation must use the pinned deployment profile')
    student_path = root / relative(read(root / BASELINE)['profile_path'])
    files_to_check.add(student_path.relative_to(root))
    student = read(student_path)
    require(step_profile['chat_template_kwargs'] == student['chat_template_kwargs'],
            'The actor and preference tokenizer must use identical chat template settings')
    tokenizer = AutoTokenizer.from_pretrained(read(root/RUNTIME)['model'],local_files_only=True,trust_remote_code=False)
    sandbox_path = collection/'private/task-runtime.json'
    budget = parent['actor_budget']
    runtime = ReplayRuntime(Sandbox(read(sandbox_path), budget['max_tool_seconds']), normalization(root), tokenizer,
                            student, step_profile, parent['agent_policy'], budget)
    plan_sha = parent['collection']['historical_protocol_sha256']
    history = root/relative(read(root/ORIGINAL_PROTOCOL)['run_directory'])
    history_audit = history/'audit.json'
    require(sha256(root/ORIGINAL_PROTOCOL)==plan_sha and sha256(history_audit)==parent['collection']['historical_audit_sha256'],
            'Historical training receipt changed')
    receipts = {row['id']: row for row in read(history_audit)['rows']}
    target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    with single_writer(target.with_suffix('.lock')):
        if target.exists():
            raise FileExistsError('Preserve the frozen first-action plan')
        rows = []
        with TemporaryDirectory(prefix='fork-prefix-',dir=target.parent) as temporary:
            for ordinal,(name,case) in enumerate(cases.items(),1):
                result = state_from_history(runtime,history,receipts[name],data,case,Path(temporary)/str(ordinal))
                rows.append(dict(result, ordinal=ordinal,task_id=name, prefix_sha256=case.proof['prefix_sha256']))
        external = [reference_path,sandbox_path,*map(Path,deployment_inputs.values())]
        plan = {'schema_version': 1, 'training_only': True, 'final_test_opened': False, 'output': str(OUTPUT),
                'collection': str(collection), 'rows': rows, 'design_sha256': sha256(DIR/'design.json'),
                'deployment_inputs':deployment_inputs,'runtime_inputs': {str(path):sha256(path) for path in external},
                'frozen_files': {str(path):sha256(root/path) for path in sorted(files_to_check)}}
        atomic_json(target,plan)
    return {'protocol_sha256':sha256(target),'training_states':len(rows),
            'families':len({case.task['family'] for case in cases.values()}),
            'historical_calls':dict(Counter(row['historical_calls'] for row in rows)),
            'prompt_tokens_min':min(row['prompt_tokens'] for row in rows),
            'prompt_tokens_max':max(row['prompt_tokens'] for row in rows),
            'generation_ceiling':2*sum(row['budget']['max_total_generated_tokens'] for row in rows),
            'rollout_ceiling':2*len(rows),'model_generations':0,'optimizer_updates':0,'final_test_opened':False}


if __name__ == '__main__':
    import os

    os.umask(0o077)
    arguments = argparse.ArgumentParser(description=__doc__)
    for name in ('root','collection','deployment','target'):
        arguments.add_argument('--'+name,type=Path,required=True)
    args = arguments.parse_args()
    print(json.dumps(freeze(args.root.resolve(),args.collection.resolve(),args.deployment.resolve(),args.target.resolve()),
                     allow_nan=False))
