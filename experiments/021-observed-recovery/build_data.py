"""Replace two slots using actual training-only states and executed corrections.

No model call occurs here. Complete histories are retained, failed actions are
masked, and only an independently verified successful repair is a target.
"""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.data.repair_curriculum import Episode, replacement_command, SLOTS
from one_gpu_lm.data.reference_episodes import repair_commands
from one_gpu_lm.data.balanced import prepare as prepare_balanced
from one_gpu_lm.evaluation.environment import RepositoryEnvironment, repository_python_site
from one_gpu_lm.evaluation.grading import generated_files, grade
from one_gpu_lm.evaluation.tasks import prepare, file_manifest


def ordered_states(states, task, slot):
    if task['split'] != 'train' or slot not in SLOTS[-2:]:
        raise ValueError('Observed recovery is training-only and restricted to two slots')
    by_call = {s['completed_calls']: s for s in states}
    if len(by_call) != len(states): raise ValueError('Duplicated observed call')
    candidates = []
    for used, state in by_call.items():
        if state['task_id'] != task['id'] or state['split'] != 'train':
            raise ValueError('Task partition or identity changed')
        if not 4 <= used <= 8 or state['terminal']: continue
        prefix = [by_call.get(i) for i in range(1, used+1)]
        if any(s is None or not s['protected_files_unchanged'] or s['source_state']['invalid_editable_files']
               or s['trajectory']['local_model'].get('backend_errors') for s in prefix): continue
        if set(state['sources']) != set(task['editable_files']): continue
        try:
            for name, content in state['sources'].items():
                if name.endswith('.py'): ast.parse(content)
        except SyntaxError:
            continue
        messages = state['trajectory']['messages']
        if not messages or messages[-1]['role'] != 'user': continue
        edited = state['source_state']['changed_editable_files'] > 0
        ever_edited = any(s['source_state']['changed_editable_files'] > 0 for s in prefix)
        if slot == 'stall_repair' and ever_edited: continue
        failed_tool = any(m.get('extra', {}).get('tool', {}).get('returncode', 0) != 0
                          for m in state['trajectory']['messages'])
        category = 'wrong_edit' if edited else 'tool_error' if failed_tool else 'read_only'
        rank = {'wrong_edit': 0, 'tool_error': 1, 'read_only': 2}[category] if slot == 'late_repair' else 0
        candidates.append((rank, -used, category, state))
    return [(category, state) for _, _, category, state in sorted(candidates, key=lambda x: x[:2])]


def within_limits(row, tokenizer, profile):
    value = encode_training_example(row, tokenizer, profile)
    return (len(value['ids']) <= profile['training']['max_sequence_length']
            and value['supervised_tokens'] <= profile['generation']['max_new_tokens'])


def build(root, collection, out):
    root, collection, out = Path(root).resolve(), Path(collection), Path(out)
    if out.exists(): raise ValueError('Preserve all data preparation attempts')
    read = lambda p: json.loads(Path(p).read_text())
    exp = root / 'experiments/021-observed-recovery'
    plan = read(exp / 'preparation-plan.json'); protocol = read(exp / 'protocol.json')
    tasks_path = root / 'private/tasks.json'; tasks = read(tasks_path)['tasks']
    training = {t['id']: t for t in tasks if t['split'] == 'train'}
    admission_path = root / 'runs/admission/admission.json'; admission = read(admission_path)
    collected = read(collection / 'collection.json')
    if (not collected['complete'] or collected['task_spec_sha256'] != sha256(tasks_path)
            or collected['protocol_sha256'] != sha256(exp / 'protocol.json')
            or len(collected['tasks']) != 55 or {r['id'] for r in collected['tasks']} != set(training)
            or any(r['split'] != 'train' for r in collected['tasks'])
            or not admission['complete'] or admission['task_spec_sha256'] != sha256(tasks_path)
            or {r['id'] for r in admission['tasks'] if r['split'] == 'train' and r['admitted']} != set(training)):
        raise ValueError('Collection/admission must exactly cover the frozen training partition')
    parent_path = root / 'runs/reference-data020/recovery/train.jsonl'
    if sha256(parent_path) != protocol['frozen_files'][str(parent_path.relative_to(root))]:
        raise ValueError('Fallback data changed')
    rows = [json.loads(line) for line in parent_path.read_text().splitlines()]
    if len(rows) != 330 or Counter(r['provenance']['task_id'] for r in rows) != Counter({k: 6 for k in training}):
        raise ValueError('Require the exact six parent slots per training task')
    for task_id in training:
        if [r['provenance']['stage'] for r in rows if r['provenance']['task_id'] == task_id] != list(SLOTS):
            raise ValueError('Parent semantic order changed')
    runtime = read(root / 'runtime.json'); profile = read(exp / 'profile-observed.json')
    from transformers import AutoTokenizer
    model = Path(runtime['model'])
    download = read(model / 'download-receipt.json')
    if any(download[k] != profile[k] for k in ['model_id', 'revision']): raise ValueError('Tokenizer identity changed')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    out.mkdir(parents=True, mode=0o700); (out / 'private').mkdir(mode=0o700)
    replacements, checks, reasons, categories = {}, [], Counter(), Counter()
    for collected_task in collected['tasks']:
        task = training[collected_task['id']]
        states = []
        for filename, digest in sorted(collected_task['states'].items()):
            path = collection / task['id'] / 'states' / filename
            if sha256(path) != digest: raise ValueError('Observed state evidence changed')
            state = read(path)
            if state['identity']['task_spec_sha256'] != sha256(tasks_path):
                raise ValueError('Observed state belongs to different tasks')
            states.append(state)
        slug = task['repository'].replace('/', '--'); snapshot = root / 'sources' / slug
        deps = root / 'node-deps' / slug / 'node_modules'
        def make_env(work):
            return RepositoryEnvironment(work, python_site=repository_python_site(root, runtime, task['repository']),
                                         node_root=runtime['node_root'], node_modules=deps if deps.exists() else None,
                                         output_chars=2400)
        for slot in SLOTS[-2:]:
            accepted = False; rejected = Counter()
            candidates = ordered_states(states, task, slot)
            for category, state in candidates:
                used = state['completed_calls']; private = out / 'private' / task['id'] / slot / str(used)
                private.mkdir(parents=True)
                workspace = prepare(snapshot, private / 'workspace', task)
                original = file_manifest(workspace, exclude=generated_files(task['repository']))
                for name, content in state['sources'].items(): (workspace / name).write_text(content)
                episode = Episode(task, workspace, snapshot, private, original, make_env)
                if episode.progress.snapshot() != state['source_state']:
                    raise ValueError('Reconstructed source differs from observed state')
                episode.messages = copy.deepcopy(state['trajectory']['messages'])
                episode.steps = [{'kind': 'observed_base_call', 'call_index': i} for i in range(1, used+1)]
                failed = grade(task, workspace, snapshot, private / 'before-grade', original, make_env)
                if failed['passed'] or failed.get('returncode') != 1:
                    rejected['state_not_independently_failed_test'] += 1; continue
                commands = repair_commands(task, workspace, snapshot)
                _, test, _, _, verify, submit = [r[1] for r in commands]
                if slot == 'late_repair':
                    episode.action('teacher_executed_failing_diagnostic', test, 1)
                    if episode.progress.snapshot() != state['source_state']:
                        raise ValueError('Diagnostic unexpectedly changed the observed source')
                try:
                    repair = replacement_command(episode.sources(), {n: (snapshot / n).read_text() for n in task['editable_files']})
                except ValueError:
                    rejected['no_bounded_unique_repair_anchor'] += 1; continue
                prefix, progress = episode.progress.prepare_request(episode.messages, len(episode.steps), 12, True)
                target = '<tool_call>\n<function=bash>\n<parameter=command>\n'+repair+'\n</parameter>\n</function>\n</tool_call>'
                if not within_limits({'messages': prefix, 'target': target}, tokenizer, profile):
                    rejected['complete_context_or_target_limit'] += 1; continue
                episode.action('correct_repair', repair, 0, slot=slot)
                episode.action('verify', verify, 0)
                episode.action('submit', submit, 0)
                produced, control = episode.finish(wrong_grade=failed)
                if len(produced) != 1 or produced[0]['target'] != target or produced[0]['messages'] != prefix:
                    raise ValueError('Verified target differs from checked complete example')
                row = produced[0]
                row['provenance'].update(curriculum='021-observed-recovery', observed_calls=used,
                    observed_state_sha256=collected_task['states']['call-%02d.json' % used],
                    state_category=category, teacher_diagnostic=slot == 'late_repair', base_actions_supervised=False)
                replacements[row['id']] = row; categories[category] += 1; accepted = True
                checks.append(dict(control, slot=slot, observed_calls=used, category=category,
                                   whole_context_preserved=True, total_calls=len(episode.steps), rejected_states=dict(rejected)))
                break
            if not accepted:
                reason = 'no_eligible_observed_state' if not candidates else 'no_verified_bounded_correction'
                reasons[reason] += 1; checks.append({'id': task['id'], 'slot': slot, 'fallback': True,
                                                    'reason': reason, 'rejected_states': dict(rejected)})
        write_json(out / 'status.json', {'complete': False, 'tasks_done': len(checks)//2, 'replaced_records': len(replacements)})
        print(task['id'], 'observed states verified', flush=True)
    result = [replacements.get(row['id'], row) for row in rows]
    output = out / 'observed'; output.mkdir(mode=0o700); data = output / 'train.jsonl'
    data.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in result))
    task_count = len({r['provenance']['task_id'] for r in replacements.values()})
    coverage = len(replacements) >= plan['data']['minimum_replaced_records'] and task_count >= plan['data']['minimum_tasks_with_replacements']
    receipt = {'schema_version': 1, 'records': 330, 'sha256': sha256(data), 'task_spec_sha256': sha256(tasks_path),
        'admission_sha256': sha256(admission_path), 'license_record_sha256': sha256(exp/'sources.json'),
        'parent_data_sha256': sha256(parent_path), 'collection_sha256': sha256(collection/'collection.json'),
        'raw_data_public': False, 'kinds': dict(Counter(r['provenance']['kind'] for r in result)),
        'task_ids': sorted(training), 'replaced_records': len(replacements), 'replaced_tasks': task_count,
        'categories': dict(categories), 'fallback_reasons': dict(reasons), 'coverage_passed': coverage,
        'checks': checks, 'evaluation_traces_used': False, 'teacher_api_used': False,
        'incorrect_actions_supervised': False, 'shared_first_four_records': 220}
    write_json(output/'receipt.json', receipt)
    prepared = root/'runs/prepared-data021/observed'
    audit = prepare_balanced({'data': data, 'data_receipt': output/'receipt.json', 'admission': admission_path,
                             'licenses': exp/'sources.json'}, profile, tokenizer, 330, prepared)
    parent_prepared = [json.loads(line) for line in (root/'runs/prepared-data020/recovery/train.jsonl').read_text().splitlines()]
    actual_prepared = [json.loads(line) for line in (prepared/'train.jsonl').read_text().splitlines()]
    if [r['id'] for r in actual_prepared] != [r['id'] for r in parent_prepared]:
        raise ValueError('Seeded semantic record order changed')
    common = [i for i, row in enumerate(parent_prepared) if row['provenance']['stage'] in SLOTS[:4]]
    if len(common) != 220 or any(actual_prepared[i] != parent_prepared[i] for i in common):
        raise ValueError('Common inspect/repair/verify/submit examples changed')
    if audit['excluded'] or audit['supervised_tokens'] < plan['data']['minimum_supervised_tokens']:
        raise ValueError('All 330 unique full examples and minimum supervision are required')
    write_json(out/'status.json', {'complete': True, 'tasks_done': 55, 'replaced_records': len(replacements),
                                  'replaced_tasks': task_count, 'coverage_passed': coverage})
    public = {k: v for k, v in receipt.items() if k not in {'checks', 'task_ids'}}
    public.update(prepared_data_sha256=sha256(prepared/'train.jsonl'), prepared_receipt_sha256=sha256(prepared/'receipt.json'),
                  selection_sha256=sha256(prepared/'selection.json'), input_tokens=audit['input_tokens'],
                  supervised_tokens=audit['supervised_tokens'], semantic_order_sha256=hashlib.sha256(
                      json.dumps([r['id'] for r in actual_prepared], separators=(',', ':')).encode()).hexdigest())
    write_json(out/'derived-data.json', public)
    return public


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'collection', 'out']: p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(build(a.root, a.collection, a.out)))
