"""Freeze and evaluate the unchanged 037 excerpt rule on independent repositories."""
import argparse
from collections import Counter
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_tasks import adapt, validate_split
from one_gpu_lm.verified.rollout import normalization
from one_gpu_lm.verified.tasks import validate_source


EXP = Path('experiments/038-independent-compact')
PRIOR = Path('experiments/037-compact-regression')
PRIVATE = Path('private/independent-compact')
DATA = PRIVATE / 'development'
OUT = Path('runs/historic038-independent-001')
ARMS = ('prefix', 'compact')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def previous(root):
    prior = str(root / PRIOR)
    if prior not in sys.path:
        sys.path.insert(0, prior)
    return load('historical037_prepare', root / PRIOR / 'prepare.py')


def schedule(tasks):
    return [
        {'task_number': n, 'task_id': task['id'], 'block':task['family'],
         'arm': arm, 'run_id': f't{n:02d}-{arm}'}
        for n, task in enumerate(tasks, 1) for arm in (ARMS if n % 2 else ARMS[::-1])
    ]


def pair_prompts(root, context, prior, task, output):
    module = load('historical036_evidence', root / prior.PARENT / 'evidence.py')
    return prior.present(task, output, normalization(root),
                         context.protocol['evaluation_budget']['max_tool_output_chars'], module.make_task)


def inputs(root):
    prior = previous(root)
    runner, older_context, _ = prior.parent_inputs(root)
    path = root / EXP / 'protocol.json'
    plan = read(path)
    original = read(root / PRIOR / 'protocol.json')
    publication = read(root / PRIVATE / 'publication.json')
    if (not publication['anonymous_files_verified'] or publication['ci_conclusion'] != 'success'
            or publication['plan_sha256'] != sha256(path) or publication['prior_report_commit'] != plan['037_commit']
            or not older_context.legacy.frozen(root, plan['frozen_files'])):
        raise ValueError('Exact public preregistration and frozen inputs required')
    same = ('model_id', 'revision', 'evaluation_budget', 'agent_policy', 'normalization',
            'recovery_accounting_seconds')
    if (any(plan[key] != original[key] for key in same)
            or plan['treatment'] != original['treatment'] | {'output_source': plan['treatment']['output_source']}
            or older_context.protocol['adapter_sha256'] or plan['optimizer_updates'] != 0
            or plan['final_test_opened'] or plan['weight_publication_allowed']):
        raise ValueError('Only the registered new development cohort may differ')
    data = root / DATA
    tasks = read(data / 'tasks.json')['tasks']
    sources = read(data / 'sources.json')['sources']
    admission = read(data / 'admission.json')
    proof = read(data / 'preflight.json')
    ids = [t['id'] for t in tasks]
    if (not admission['complete'] or admission['model_calls'] or admission['final_test_opened']
            or not proof['passed'] or proof['model_calls'] or proof['final_test_opened']
            or ids != [r['id'] for r in admission['rows']] or not all(r['admitted'] for r in admission['rows'])
            or len(ids) != len(set(ids)) or len(ids) != plan['tasks']):
        raise ValueError('Complete score-blind admission required')
    validate_split(sources, admission['previous_families'])
    source_map = {s['source_id']: s for s in sources}
    for task in tasks:
        source = source_map[task['source_id']]
        validated, _ = adapt(source, task)
        validate_source(validated, data / 'snapshots' / source['source_id'], 'development')
    if (Counter(t['family'] for t in tasks) != plan['families']
            or Counter(source_map[t['source_id']]['language'] for t in tasks) != plan['language_counts']):
        raise ValueError('Registered families, task counts or languages differ')
    order = schedule(tasks)
    if (len(order) != plan['rollouts'] or fingerprint(order) != plan['schedule_sha256']
            or proof['schedule_sha256'] != plan['schedule_sha256']
            or prior.compact_counts(proof) != plan['preflight']):
        raise ValueError('Registered task order or prompt preflight differs')
    context = replace(older_context, plan_path=path, plan=plan, data=data, tasks=dict(zip(ids, tasks)),
                      sources=source_map, order=order, runtime_path=root / PRIVATE / 'offline-runtime.json')
    return runner, context, proof, prior


def variants(root, context, proof, prior):
    result = {}
    evidence = {r['task_number']: r for r in proof['rows']}
    for item in context.order[::2]:
        number = item['task_number']
        path = context.data / 'public-tests' / f't{number:02d}.json'
        row = evidence[number]
        if sha256(path) != row['transcript_sha256']:
            raise ValueError('Opening tool evidence changed')
        prefix, compact, stats = pair_prompts(root, context, prior, context.tasks[item['task_id']], read(path))
        if any(row[key] != value for key, value in stats.items()):
            raise ValueError('Compaction statistics differ from preflight')
        for arm, variant in zip(ARMS, (prefix, compact)):
            if prior.text_hash(variant['description']) != row['description_sha256'][arm]:
                raise ValueError('Opening prompt changed')
            result[f't{number:02d}-{arm}'] = variant
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    runner, context, proof, prior = inputs(root)
    prompts = variants(root, context, proof, prior)
    if arguments.check:
        print(json.dumps({'tasks': len(context.tasks), 'rollouts': len(prompts), 'final_test_opened': False}))
        return

    def perform(root, data, task, source, backend, sandbox, protocol, directory, arm, **kwargs):
        selected = prompts[directory.name]
        if selected['id'] != task['id']:
            raise ValueError('Misbound scheduled task')
        return runner.rollout(root, data, selected, source, backend, sandbox, protocol, directory, arm, **kwargs)

    runner.sweep(root, root / OUT, context, perform=perform)


if __name__ == '__main__':
    main()
