"""Freeze, train and audit one full-continuation LoRA candidate."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import signal
from tempfile import TemporaryDirectory

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import identity_digest, single_writer
from one_gpu_lm.common.training_encoding import encode_training_example

from .continuation_training import audit, json_bytes, train, unchanged
from .continuation_training_data import BASELINE, BASELINE_AUDIT, BASELINE_REVIEW, SECRET_SHAPES
from .continuation_training_data import load_baseline, matched_windows, read, record_shape, rows
from .decision_state_collection import require
from .recovery_sft_projection import ORIGINAL_PROTOCOL, TARGET_KIND, supplements
from .tasks import digest, relative


EXPERIMENT = '044-deployment-recovery-sft'
EXPERIMENT_DIR = Path('experiments') / EXPERIMENT
DESIGN = EXPERIMENT_DIR / 'design.json'
INPUTS = Path('private') / EXPERIMENT
PROTOCOL = INPUTS / 'protocol.json'
PROJECTED = INPUTS / 'projected.jsonl'
REPLAY = INPUTS / 'replay.json'
MIX = INPUTS / 'mixture.json'
DATA = INPUTS / 'data.jsonl'
OUTPUT = Path('runs') / 'q044-recovery-sft'


def load(root):
    from transformers import AutoTokenizer

    runtime = read(root/'private/parent-runtime.json')
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'], local_files_only=True, trust_remote_code=False)
    baseline = load_baseline(root, tokenizer)
    design = read(root/DESIGN)
    require(design['experiment'] == EXPERIMENT and design['final_test_opened'] is False
            and design['weights_public'] is False
            and (design['model_id'], design['base_revision']) == (baseline.profile['model_id'], baseline.profile['revision'])
            and design['retained_training']['records'] == len(baseline.raw)
            and design['retained_training']['optimizer_updates'] == baseline.baseline['optimizer_updates'],
            'Training design or retained profile differs from the pinned comparison')
    return runtime, tokenizer, baseline, design


def independent_development(root, baseline, design, cohort_path):
    spec = design['development']
    require(sha256(cohort_path) == spec['cohort_sha256'], 'The prepared independent cohort changed')
    cohort = read(cohort_path)
    source_path = cohort_path.with_name('sources.json')
    require(cohort['model_generations'] == 0 and cohort['final_test_opened'] is False
            and cohort['cases'] == spec['tasks'] and len(cohort['families']) == spec['families']
            and sha256(source_path) == cohort['sources_sha256'], 'Expected the score-free independent development inventory')
    train_sources = read(root/relative(baseline.protocol['sources_path']))['sources']
    dev_sources = read(source_path)['sources']

    def families(sources):
        return {name.casefold().split(':', 1)[0] for source in sources
                for name in (source['repository'], source['family'], *source.get('related_families', []))}

    require(len(dev_sources) == spec['tasks'] and all(item['split'] == 'development' for item in dev_sources)
            and not families(train_sources).intersection(families(dev_sources)), 'Training and development repository families overlap')


def encode_projected(records, baseline, tokenizer, design):
    accepted, groups = [], defaultdict(list)
    seen = set()
    old_tasks = {row['provenance']['task_id'] for row in baseline.raw}
    for row in records:
        v = row['provenance']
        proof = row['encoding_receipt']
        require(v['split'] == 'train' and v['producer_kind'] == 'model' and v['executed'] is True
                and v['task_id'] not in old_tasks and v['repository'] in baseline.review['sources']
                and v['collection_audit_sha256'] == design['collection']['audit_sha256']
                and v['source_manifest_sha256'] == baseline.baseline['source_manifest_sha256']
                and v['target_transform'] == TARGET_KIND, 'New label left its reviewed training provenance')
        episode = identity_digest([TARGET_KIND, v['task_id'], v['trajectory_sha256']])
        require(row['episode_id'] == episode and row['id'] == identity_digest([episode,v['call']]), 'Action ID is not bound to the reviewed trajectory')
        encoded = encode_training_example(row, tokenizer, baseline.profile)
        encoding = {'tokens': len(encoded['ids']), 'target_tokens': encoded['supervised_tokens'],
                    'encoding_sha256': identity_digest(encoded['ids'])}
        require(encoding == proof and proof['encoding_sha256'] not in seen
                and len(encoded['ids']) <= baseline.profile['training']['max_sequence_length']
                and 0 < encoded['supervised_tokens'] <= design['actor_budget']['max_generated_tokens_per_call'],
                'Replayed action encoding changed, exceeded the budget or was duplicated')
        seen.add(proof['encoding_sha256'])
        groups[episode].append(v)
        accepted.append(dict(encoded, id=row['id'], episode_id=episode, record_group_kind=TARGET_KIND))
    for group in groups.values():
        calls, kinds = [r['call'] for r in group], [r['kind'] for r in group]
        require(calls == sorted(set(calls)) and kinds.count('submit') == 1 and kinds[-1] == 'submit'
                and 'edit' in kinds and 'regression_pass' in kinds, 'Partial or reordered recovery sequence')
    require(len(groups) == len({row['provenance']['task_id'] for row in records}), 'Require one continuation per distinct training task')
    return accepted


def mixture(baseline, projected, encoded, replay):
    windows = matched_windows(baseline.encoded, encoded, baseline.recipe['gradient_accumulation_records'])
    indexed = {row['id']: row for row in [*baseline.raw, *projected]}
    ordered = [indexed[row['id']] for batch in windows for row in batch]
    groups = Counter(r['record_group_kind'] for batch in windows for r in batch)
    require(len(ordered) == len(indexed) == len(baseline.raw) + len(projected)
            and len(windows) == baseline.baseline['optimizer_updates'], 'Training mixture changed the retained records or update count')
    require(len(encoded) == len(projected) == sum(row['records'] for row in replay['outcomes'] if row['selected']),
            'A verified continuation was omitted or changed after its recorded replay')
    summary = {'retained_031': baseline.baseline, 'new_records': len(projected),
               'new_tasks': len({row['episode_id'] for row in projected}),
               'new_source_families': len({row['provenance']['family'] for row in projected}),
               'new_action_kinds': dict(Counter(row['provenance']['kind'] for row in projected)),
               'new_input_tokens': sum(len(row['ids']) for row in encoded),
               'new_supervised_tokens': sum(row['supervised_tokens'] for row in encoded),
               'input_tokens': sum(len(row['ids']) for batch in windows for row in batch),
               'supervised_tokens': sum(row['supervised_tokens'] for batch in windows for row in batch),
               'max_sequence_tokens': max(len(row['ids']) for batch in windows for row in batch),
               'max_new_target_tokens': max(row['supervised_tokens'] for row in encoded),
               'optimizer_updates': len(windows), 'record_counts': dict(groups), 'window_sizes': list(map(len, windows)),
               'batch_identity_sha256': identity_digest([[record_shape(row) for row in batch] for batch in windows]),
               'target_truncation': False, 'oversampling': False, 'final_test_opened': False, 'weights_public': False}
    return ordered, windows, summary


def scan(records, needles):
    patterns = tuple(re.compile(pattern) for pattern in SECRET_SHAPES)
    for record in records:
        text = json.dumps(record, ensure_ascii=False)
        if any(pattern.search(text) for pattern in patterns) or any(value and value in text for value in needles):
            raise ValueError('A proposed sample contains a credential pattern or a known private identifier')


def jsonl(records):
    return ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in records).encode()


def freeze(root, collection, cohort, needles=()):
    with single_writer(root/'private/044-freeze.lock'):
        require(not (root/INPUTS).exists() and not (root/OUTPUT).exists(), 'Use one immutable training registration')
        _, tokenizer, baseline, design = load(root)
        independent_development(root, baseline, design, cohort)
        projected, replay = supplements(root, collection, baseline, tokenizer, design)
        scan(projected, needles)
        encoded = encode_projected(projected, baseline, tokenizer, design)
        ordered, _, summary = mixture(baseline, projected, encoded, replay)
        origin = baseline.protocol
        values = {PROJECTED: jsonl(projected), REPLAY: json_bytes(replay), MIX: json_bytes(summary), DATA: jsonl(ordered)}
        names = {BASELINE, BASELINE_REVIEW, BASELINE_AUDIT, ORIGINAL_PROTOCOL, DESIGN, EXPERIMENT_DIR/'run.py',
                 Path('private/parent-runtime.json'), Path('experiments/023-evaluation-stability/normalize.py'),
                 Path('experiments/024-feedback-noise/normalize.py'), Path('experiments/024-feedback-noise/_feedback024.py'),
                 Path(origin['run_directory'])/'result.json', Path(origin['run_directory'])/'selection.json'}
        names.update(Path(origin[k]) for k in ('profile_path','cycle_spec','sources_path','data_path','export_manifest',
                                               'model_manifest_path','environment_path'))
        names.update(Path(name) for name in origin['correction_mix']['frozen_parent_files'])
        names.update(path.relative_to(root) for path in (root/'one_gpu_lm').rglob('*.py'))
        frozen = {str(name): sha256(root/name) for name in sorted(names)}
        frozen.update({str(name): digest(payload) for name,payload in values.items()})
        plan = {'schema_version': 1, 'kind': TARGET_KIND, 'candidate': EXPERIMENT, 'model_id': design['model_id'],
                'revision': design['base_revision'], 'run_directory': str(OUTPUT), 'data_path': str(DATA),
                **{k: origin[k] for k in ('profile_path','sources_path','model_manifest_path','environment_path','expected_trainable_parameters')},
                'expected_optimizer_updates': summary['optimizer_updates'], 'expected_records': len(ordered),
                'collection': design['collection'], 'development_cohort_sha256': design['development']['cohort_sha256'],
                'data_mixture_sha256': identity_digest(summary), 'frozen_files': frozen,
                'final_test_opened': False, 'weights_public': False}
        values[PROTOCOL] = json_bytes(plan)
        with TemporaryDirectory(prefix='.freeze-recovery-', dir=root/'private') as temporary:
            staged = Path(temporary)/'training'
            staged.mkdir(mode=0o700)
            for name, content in values.items():
                (staged/name.name).write_bytes(content)
            staged.rename(root/INPUTS)
    return {'private_protocol_sha256': sha256(root/PROTOCOL), 'design_sha256': sha256(root/DESIGN),
            'frozen_files': len(frozen), 'trainable_parameters': plan['expected_trainable_parameters'],
            'records': len(ordered), 'summary': summary, 'replay_counts': replay['counts'],
            'rejected_replay_cases': dict(Counter(item['reason'] for item in replay['outcomes'] if not item['replay_passed'])),
            'optimizer_updates_completed': 0, 'final_test_opened': False, 'weights_public': False}


def registered(root, publication_path):
    protocol = root/PROTOCOL
    plan, checksum, publication = read(protocol), sha256(protocol), read(publication_path)
    paths = {str(name) for name in (DESIGN, PROJECTED, REPLAY, MIX, DATA)}
    require(plan['candidate'] == EXPERIMENT and plan['kind'] == TARGET_KIND
            and plan['run_directory'] == str(OUTPUT) and plan['data_path'] == str(DATA)
            and plan['final_test_opened'] is plan['weights_public'] is False
            and publication['protocol_sha256'] == checksum and publication['ci_conclusion'] == 'success'
            and publication['anonymous_files_verified'] is True and re.fullmatch(r'[0-9a-f]{40}', publication['commit'])
            and paths <= set(plan['frozen_files']) and unchanged(root, plan['frozen_files']),
            'Require verified frozen inputs and the published training registration')
    runtime, tokenizer, baseline, design = load(root)
    projected, replay = rows(root/PROJECTED), read(root/REPLAY)
    require(replay['new_model_generations'] == 0 and replay['final_test_opened'] is False,
            'Training preparation exceeded the registered evidence scope')
    encoded = encode_projected(projected, baseline, tokenizer, design)
    raw, batches, summary = mixture(baseline, projected, encoded, replay)
    require(all(plan[key] == baseline.protocol[key] for key in ('profile_path','sources_path','model_manifest_path',
                                                                 'environment_path','expected_trainable_parameters'))
            and raw == rows(root/DATA) and summary == read(root/MIX)
            and plan['data_mixture_sha256'] == identity_digest(summary)
            and plan['collection'] == design['collection']
            and plan['development_cohort_sha256'] == design['development']['cohort_sha256']
            and plan['expected_optimizer_updates'] == len(batches) and plan['expected_records'] == len(raw)
            and (plan['model_id'],plan['revision']) == (baseline.profile['model_id'],baseline.profile['revision'])
            and sha256(protocol) == checksum, 'Reconstructed training data differs from its preregistration')
    added = Counter(row['provenance']['repository'] for row in projected)
    sources = {name: dict(entry, verified_fragment_records=entry['verified_fragment_records'] + added[name])
               for name, entry in baseline.review['sources'].items()}
    review = dict(baseline.review, sources=sources)
    return plan, checksum, runtime, replace(baseline, review=review), raw, batches, summary


def main():
    os.umask(0o077)
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, required=True)
    commands = cli.add_subparsers(dest='mode', required=True)
    prep = commands.add_parser('freeze')
    prep.add_argument('--collection', type=Path, required=True)
    prep.add_argument('--development-cohort', type=Path, required=True)
    prep.add_argument('--private-identifiers', type=Path)
    for mode in ('train', 'audit'):
        commands.add_parser(mode).add_argument('--publication', type=Path, required=True)
    args = cli.parse_args()
    root = args.root.resolve()
    if args.mode == 'freeze':
        needles = read(args.private_identifiers) if args.private_identifiers else ()
        result = freeze(root, args.collection.resolve(), args.development_cohort.resolve(), needles)
    else:
        plan, checksum, runtime, baseline, raw, batches, summary = registered(root, args.publication.resolve())
        if args.mode == 'train':
            def interrupted(*_):
                raise KeyboardInterrupt('Training interrupted')

            for kind in (signal.SIGTERM, signal.SIGINT):
                signal.signal(kind, interrupted)
            result = train(root, plan, checksum, runtime, baseline, raw, batches, summary)
            require(result['complete'], 'No complete eligible training candidate')
        else:
            result = audit(root, plan, checksum, baseline, raw, batches, summary)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
