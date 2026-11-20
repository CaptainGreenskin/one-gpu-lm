"""Select CPU-admitted cases and record their exact, score-blind openings."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
import shutil
import tempfile

from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_tasks import prepare, validate_split
from one_gpu_lm.verified.historic_policy import initial_messages

from study import ARMS, DATA, EXP, OUT, PRIOR, PRIVATE, pair_prompts, previous, read, schedule


def select(root, plan):
    private = root / PRIVATE
    candidates = private / 'candidates'
    review = read(private / 'lineage-review.json')
    if (not review['passed'] or review['final_test_opened'] or review['overlap']
            or review['prior_report_commit'] != plan['037_commit']):
        raise ValueError('Require the reviewed development-only family inventory')
    index = read(private / 'admission-index.json')
    runtime = private / 'offline-runtime.json'
    runtime_hash = sha256(runtime)
    mounted = {item['sha256'] for item in read(runtime)['mounts']}
    selections = {}
    counts = {}
    for family, size in plan['families'].items():
        stem = family.replace('/', '--')
        directory = private / f"admission-{stem}-{index[family]}"
        status = read(directory / 'status.json')
        task_path = candidates / f'{stem}-tasks.json'
        source_path = candidates / f'{stem}-sources.json'
        if (not status['complete'] or status['model_calls'] or status['final_test_opened']
                or status['family'] != family or status['tasks_sha256'] != sha256(task_path)
                or status['sources_sha256'] != sha256(source_path) or status['runtime_sha256'] != runtime_hash):
            raise ValueError('Require complete CPU-only candidate admission')
        source = {s['source_id']: s for s in read(source_path)['sources']}
        tasks = {t['id']: t for t in read(task_path)['tasks']}
        rows = sorted((r for r in status['rows'] if r['admitted']), key=lambda r: fingerprint(r['id']))
        if len(rows) < size:
            raise ValueError(f'Insufficient CPU-admitted tasks for {family}')
        selections[family] = [(tasks[r['id']], source[tasks[r['id']]['source_id']], directory / r['id'], r)
                              for r in rows[:size]]
        for task, source, _, _ in selections[family]:
            reference = {source['files'][path] for path in task['editable_files']}
            broken = {hash_bytes(m['after'].encode()).hexdigest() for m in task['mutations']}
            if mounted & (reference | broken):
                raise ValueError('Target implementation is present in the test-only runtime')
        counts[family] = {'candidates': len(status['rows']), 'admitted': len(rows), 'selected': size,
                          'status_sha256': sha256(directory / 'status.json')}
    families = sorted(selections)
    ordered = [selections[family][position] for position in range(len(selections[families[0]]))
               for family in (families if position % 2 == 0 else families[::-1])]
    validate_split([source for _, source, _, _ in ordered], review['previous_families'])
    return ordered, counts, review['previous_families']


def freeze(root):
    path = root / EXP / 'protocol.json'
    if path.exists() or (root / DATA).exists() or (root / OUT).exists():
        raise ValueError('Preserve the first frozen selection and protocol')
    design = read(root / EXP / 'design.json')
    prior = previous(root)
    _, context, _ = prior.parent_inputs(root)
    original = read(root / PRIOR / 'protocol.json')
    selected, admission_counts, previous_families = select(root, design)
    tasks = [t for t, _, _, _ in selected]
    sources = [s for _, s, _, _ in selected]
    data = root / DATA
    language = Counter(s['language'] for s in sources)
    if len(tasks) != design['tasks'] or language != design['language_counts']:
        raise ValueError('Cohort count or language coverage differs')
    private = root / PRIVATE
    policy = context.protocol['agent_policy']
    budget = context.protocol['evaluation_budget']
    profile = read(root / context.protocol['evaluation_profile_path'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(context.runtime['model'], local_files_only=True, trust_remote_code=False)

    def encode(messages):
        return encode_prefix(tokenizer, messages, profile['chat_template_kwargs'], policy['format'])[0]

    rows = []
    results = []
    with tempfile.TemporaryDirectory(prefix='preflight-', dir=private) as temp:
        stage = Path(temp) / 'development'
        for name in ('snapshots', 'public-tests'):
            (stage / name).mkdir(parents=True, exist_ok=True)
        for number, (task, source, case, row) in enumerate(selected, 1):
            checks = {r['label']: r for r in row['checks']}
            required = ('parent-0', 'parent-1', 'fixed-0', 'fixed-1', 'grader')
            for label in required:
                if not checks[label]['passed'] or checks[label]['sha256'] != sha256(case / f'{label}.json'):
                    raise ValueError('Original CPU test receipt changed')
            sid = task['source_id']
            snapshot = stage / 'snapshots' / sid
            shutil.copytree(private / 'candidates' / 'snapshots' / sid, snapshot)
            before = prepare(snapshot, Path(temp) / f'original-{number:02d}', source, task)
            for name, digest in before.items():
                if sha256(case / 'parent-0' / name) != digest:
                    raise ValueError('CPU public output does not match the original workspace')
            output = case / 'parent-0.json'
            shutil.copyfile(output, stage / 'public-tests' / f't{number:02d}.json')
            prefix, compact, stats = pair_prompts(root, context, prior, task, read(output))
            measurements = {}
            for arm, variant in zip(ARMS, (prefix, compact)):
                messages = initial_messages(variant, policy, budget)
                ids, _ = prepare_context(messages, encode, budget['max_input_tokens'], policy['context'])
                if not ids or len(ids) > budget['max_input_tokens']:
                    raise ValueError('Opening exceeds per-call input cap')
                measurements[arm] = (len(ids), prior.text_hash(json.dumps(ids, separators=(',', ':'))),
                                     prior.text_hash(variant['description']))
            stats.update(task_number=number, transcript_sha256=sha256(output),
                         opening_tokens={a: v[0] for a, v in measurements.items()},
                         opening_token_sha256={a: v[1] for a, v in measurements.items()},
                         description_sha256={a: v[2] for a, v in measurements.items()})
            row.update(workspace_sha256=fingerprint(sorted(before.items())))
            rows.append(stats)
            results.append(row)
        order = schedule(tasks)
        proof = dict(passed=True, tasks=len(tasks), model_calls=0, new_tool_calls=0, final_test_opened=False,
                     schedule_sha256=fingerprint(order), rows=rows)
        write_json(stage / 'tasks.json', {'tasks':tasks})
        write_json(stage / 'sources.json', {'sources':sources})
        write_json(stage / 'admission.json', dict(complete=True,model_calls=0,final_test_opened=False,rows=results,
                                                previous_families=previous_families))
        write_json(stage / 'preflight.json', proof)
        stage.rename(data)
    unchanged = ('model_id', 'revision', 'evaluation_budget', 'agent_policy', 'normalization',
                 'recovery_accounting_seconds', 'primary_outcome', 'arms', 'optimizer_updates',
                 'final_test_opened', 'weight_publication_allowed')
    frozen = [EXP / name for name in ('design.json', 'freeze.py', 'study.py', 'audit_fresh.py', 'report.py')]
    frozen += [PRIOR / name for name in ('protocol.json', 'compact_feedback.py', 'prepare.py', 'audit.py')]
    frozen += [Path(name) for name in original['frozen_files'] if not name.startswith(('private/', 'runs/'))]
    frozen += [PRIVATE / name for name in ('offline-runtime.json', 'runtime-proof.json', 'lineage-review.json',
                                         'admission-index.json', 'admit_remote.py')]
    frozen += [DATA / name for name in ('tasks.json', 'sources.json', 'admission.json', 'preflight.json')]
    frozen += [DATA / 'public-tests' / f't{i:02d}.json' for i in range(1, len(tasks) + 1)]
    treatment = dict(original['treatment'], output_source='New score-blind CPU receipt from each admitted parent.')
    plan = dict(design, **{k: original[k] for k in unchanged}, treatment=treatment,
                frozen_utc=datetime.now(timezone.utc).isoformat(), schedule_sha256=proof['schedule_sha256'],
                rollouts=len(order), admission=admission_counts, preflight=prior.compact_counts(proof),
                frozen_files={str(name): sha256(root / name) for name in sorted(set(frozen))})
    write_json(path, plan)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    value = freeze(args.root.resolve())
    print(json.dumps({k: value[k] for k in ('rollouts', 'admission', 'preflight', 'schedule_sha256')}))
