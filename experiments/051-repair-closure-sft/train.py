"""Train and independently audit one base-initialized LoRA on verified closure actions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, identity_digest
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.models.release_lineage import audited_training
from one_gpu_lm.verified import continuation_training as trainer
from one_gpu_lm.verified.candidate import validate_recipe
from one_gpu_lm.verified.candidate_data import windows
from one_gpu_lm.verified.runtime_identity import verify as verify_environment
from one_gpu_lm.verified.tasks import relative


EXP = Path(__file__).resolve().parent
JOB = Path('private/051-repair-closure-sft')
RUN = Path('runs/q051-repair-closure-sft')
COLLECTION = Path('private/050-verified-repair-closure-v2')
PREFLIGHT = Path('private/051-closure-candidate-preflight')
DATA = Path('runs/q050-verified-repair-closure-v2/private/data.jsonl')


def read(path):
    return json.loads(path.read_bytes())


def require(valid, message):
    if not valid:
        raise ValueError(message)


def fixed(path, value):
    if path.is_file():
        require(read(path) == value, 'Keep the first frozen object')
    else:
        atomic_json(path, value)


def source_manifest(root, records):
    collect = read(root / COLLECTION / 'protocol.json')
    runner = root / COLLECTION / 'code/run.py'
    require(sha256(runner) == collect['files'][str(COLLECTION / 'code/run.py')],
            'Approved source reader changed')
    module = runpy.run_path(str(runner))
    cases, _ = module['inputs'](root, collect)
    found = {case.source['source_id']: case.source for case in cases}
    counts = Counter(record['provenance']['source_id'] for record in records)
    inventory = []
    for key, count in sorted(counts.items()):
        source = found[key]
        require(source['split'] == 'training' and source['approved_for']['training']
                and source['notice_review_complete'] and source['lineage_review_complete'],
                'A training source has no recorded review')
        inventory.append({'source_id': key, 'repository': source['repository'], 'family': source['family'],
                          'revision': source['revision'], 'parent': source['upstream_parent'],
                          'license': source['license'], 'records': count,
                          'notices': {path: source['files'][path] for path in source['notices']},
                          'allow_weight_publication': source['approved_for']['weight_publication'],
                          'aliases': source.get('related_families', [])})
    return inventory


def materialize(root, plan):
    from transformers import AutoTokenizer

    profile = read(root / relative(plan['profile_path']))
    spec = read(root / relative(plan['cycle_spec']))
    runtime = read(root / relative(plan['runtime_path']))
    verify_environment(read(root / relative(plan['environment_path'])))
    validate_recipe(profile, spec)
    recipe = spec['candidate']
    require((profile['model_id'], profile['revision']) == (plan['model_id'], plan['revision'])
            and all(recipe[name] == value for name, value in plan['recipe'].items()),
            'Pinned model and training profile differ')
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'], local_files_only=True, trust_remote_code=False)
    data = root / relative(plan['data_path'])
    records = [json.loads(line) for line in data.read_bytes().splitlines()]
    by_id = {row['id']: row for row in records}
    schedule = read(root / relative(plan['schedule_path']))
    order = sorted(by_id, key=lambda key: hashlib.sha256(('051-repair-closure-v1\0' + key).encode()).hexdigest())
    require(len(records) == len(order) == plan['expected_records'] and identity_digest(order) == plan['order_sha256']
            and [row['id'] for row in schedule] == order, 'Frozen sample order or population changed')
    manifest = root / relative(plan['sources_path'])
    source = {row['source_id']: row for row in read(manifest)['sources']}
    require(sum(row['records'] for row in source.values()) == len(records), 'Frozen source counts differ')
    raw = []
    encoded = []
    counts = Counter()
    for scheduled in schedule:
        record = by_id[scheduled['id']]
        prov = record['provenance']
        item = source[prov['source_id']]
        require(prov['split'] == 'train' and prov['executed'] and not prov['model_generated']
                and prov['family'] == item['family'], 'Record provenance differs from the audited source')
        row = encode_training_example(record, tokenizer, profile)
        prefix = row['prefix_tokens']
        require(identity_digest(row['labels']) == record['encoding_sha256'] == scheduled['labels_sha256']
                and len(row['ids']) == scheduled['token_count'] <= profile['training']['max_sequence_length']
                and row['supervised_tokens'] == scheduled['supervised_token_count']
                and row['labels'][:prefix] == [-100] * prefix and row['labels'][prefix:] == row['ids'][prefix:],
                'Assistant-only labels or token lengths changed')
        counts[record['record_group_kind']] += 1
        raw.append(dict(record, provenance=dict(prov, repository=item['repository'])))
        encoded.append({**{key: record[key] for key in ('id', 'episode_id', 'record_group_kind')}, **row})
    batches = list(windows(encoded, recipe['gradient_accumulation_records']))
    summary = {'records': len(raw), 'episodes': len({row['episode_id'] for row in raw}),
               'input_tokens': sum(len(row['ids']) for row in encoded),
               'supervised_tokens': sum(row['supervised_tokens'] for row in encoded),
               'target_counts': dict(counts), 'optimizer_updates': len(batches)}
    require(summary == plan['selection'], 'Audited token and training action counts changed')
    by_repo = {}
    for entry in source.values():
        repo = by_repo.setdefault(entry['repository'], {'license': entry['license'],
                                  'revisions': [], 'used_records': 0, 'verified_fragment_records': 0,
                                  'recorded_weight_publication_approval': True})
        require(repo['license'] == entry['license'], 'Distinct source licenses require explicit review')
        repo['revisions'].append({key: entry[key] for key in ('revision', 'parent', 'notices')})
        repo['used_records'] += entry['records']
        repo['recorded_weight_publication_approval'] &= entry['allow_weight_publication']
    review = {'sources': by_repo, 'source_manifest_sha256': sha256(manifest),
              'model_id': profile['model_id'], 'revision': profile['revision'],
              'source_snapshots_match': True, 'all_source_notices_and_lineage_reviewed_for_training': True,
              'audited_checkpoint_matches': True, 'eligible_for_existing_scope_of_development': True,
              'final_test_opened': False, 'weights_public': False}
    return runtime, SimpleNamespace(profile=profile, recipe=recipe, review=review), raw, batches, summary


def freeze(root):
    design_path = EXP / 'design.json'
    design = read(design_path)
    approved = root / PREFLIGHT / 'audit.json'
    proof = read(approved)
    collect = root / COLLECTION / 'protocol.json'
    require(sha256(collect) == proof['protocol_sha256'] == design['collection_protocol_sha256']
            and proof['status_sha256'] == design['collection_status_sha256']
            and proof['schedule_order_sha256'] == design['schedule_order_sha256']
            and proof['source_and_notice_reviewed'] and not proof['final_test_opened']
            and proof['optimizer_updates'] == proof['model_generations'] == 0, '050 data handoff was not verified')
    data = root / DATA
    schedule_path = PREFLIGHT / 'schedule.json'
    require(sha256(data) == proof['export_sha256'] and sha256(root/schedule_path) == proof['schedule_sha256'],
            '050 labels or frozen order changed')
    records = [json.loads(line) for line in data.read_bytes().splitlines()]
    job = root / JOB
    sources = JOB / 'sources.private.json'
    fixed(root / sources, {'sources': source_manifest(root, records), 'final_test_opened': False})
    cfg = {'data_path': str(DATA), 'sources_path': str(sources), 'schedule_path': str(schedule_path),
           'cycle_spec': str(PREFLIGHT / 'cycle-spec.json'), 'runtime_path': 'private/parent-runtime.json',
           **{key: design[key] for key in ('profile_path', 'environment_path', 'model_manifest_path')}}
    frozen = list((root/'one_gpu_lm').rglob('*.py'))
    frozen += [root / relative(name) for name in cfg.values()]
    frozen += [EXP / 'train.py', design_path, approved, collect]
    plan = {'schema_version': 1, 'candidate': design['experiment'], 'kind': 'verified_repair_closure_lora',
            'model_id': design['model_id'], 'revision': design['revision'],
            'run_directory': str(RUN), 'order_sha256': proof['schedule_order_sha256'],
            'expected_trainable_parameters': design['trainable_parameters'],
            'expected_records': proof['reencoded_records'], 'selection': design['selection'], 'recipe': design['recipe'],
            **cfg, 'final_test_opened': False, 'weights_public': False,
            'frozen_files': {str(path.relative_to(root)):sha256(path) for path in sorted(set(frozen))}}
    require(not (root / RUN).exists(), 'Preserve the first GPU attempt')
    _, _, _, _, summary = materialize(root, plan)
    require(summary['records'] == proof['reencoded_records']
            and summary['optimizer_updates'] == proof['expected_optimizer_updates']
            and summary['input_tokens'] == proof['input_tokens']
            and summary['supervised_tokens'] == proof['supervised_tokens'], 'CPU handoff differs from 050')
    path = job / 'protocol.json'
    fixed(path, plan)
    return {'protocol_sha256': sha256(path), 'design_sha256': sha256(design_path),
            'source_review_sha256': plan['frozen_files'][str(sources)],
            'schedule_sha256': proof['schedule_sha256'], 'order_sha256': plan['order_sha256'],
            'selection': summary, 'optimizer_updates': 0, 'final_test_opened': False}


def execute(root, phase):
    job = root / JOB
    path = job / 'protocol.json'
    plan = read(path)
    checksum = sha256(path)
    approved = read(job / 'publication.json')
    require(approved['protocol_sha256'] == checksum and approved['anonymous_files_verified'] is True
            and approved['ci_conclusion'] == 'success' and trainer.unchanged(root, plan['frozen_files']),
            'Require the unchanged executable registration and public receipt')
    runtime, baseline, raw, batches, summary = materialize(root, plan)
    if phase == 'train':
        return trainer.train(root, plan, checksum, runtime, baseline, raw, batches, summary)
    evidence = trainer.audit(root, plan, checksum, baseline, raw, batches, summary)
    run = root / RUN
    audited_training(run/'candidate-input-review.json', path, root/plan['profile_path'], run/'integrity.json')
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'train', 'audit'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    value = freeze(args.root.resolve()) if args.phase == 'freeze' else execute(args.root.resolve(), args.phase)
    print(json.dumps(value, allow_nan=False), flush=True)
