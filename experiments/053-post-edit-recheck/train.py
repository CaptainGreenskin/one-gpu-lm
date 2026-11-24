"""Train one adapter on the original actions plus verified post-edit decisions."""

import argparse
from collections import Counter
from hashlib import sha256 as hash_bytes
import json
import os
from pathlib import Path
import runpy
from tempfile import NamedTemporaryFile

from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import identity_digest, sync_directory
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.models.release_lineage import audited_training
from one_gpu_lm.verified import continuation_training as trainer
from one_gpu_lm.verified.historic_admit import semantic_failure


EXP = Path('experiments/053-post-edit-recheck')
JOB = Path('private/053-post-edit-recheck-training')
OUT = Path('runs/q053-post-edit-recheck-sft')
PREVIOUS = Path('private/052-grounded-first-edit-training/protocol.json')
REPLAY = Path('runs/q052-grounded-first-edit')
PILOT = Path('runs/q053-post-edit-recheck-preflight')
PILOT_PROTOCOL = Path('private/053-post-edit-recheck-preflight/protocol.json')
MATERIALIZER = Path('experiments/051-repair-closure-sft/train.py')


def read(path):
    return json.loads(path.read_bytes())


def require(valid, detail):
    if not valid:
        raise ValueError(detail)


def put(path, data):
    if not path.exists():
        with NamedTemporaryFile(dir=path.parent, prefix='.frozen-') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(stream.name, path)
            except FileExistsError:
                pass
            sync_directory(path.parent)
    require(path.is_file() and not path.is_symlink() and path.read_bytes() == data,
            'Preserve the first frozen training bytes')


def json_lines(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows).encode()


def seed_targets(root, original):
    """Restore the masked decisions from their next executed action's prefix."""
    groups = {row['episode_id']: [] for row in original}
    for row in original:
        groups[row['episode_id']].append(row)
    result = []
    for episode, records in groups.items():
        folder = root / REPLAY / 'private' / episode.rsplit(':', 1)[1]
        trace_path = folder / 'actor-trace.json'
        trace = read(trace_path)
        for step in trace['steps']:
            if step['kind'] != 'partial_repair_regression':
                continue
            call = step['call']
            tool = folder / f'tool-{call:03d}.json'
            require(sha256(tool) == step['tool_sha256'] and semantic_failure(read(tool))
                    and step['source'] not in (trace['original_source'], trace['final_source'])
                    and not step['changed'] and read(folder/'grade.json')['passed'],
                    'The intermediate regression must fail on a real partial patch')
            following = next(row for row in records if row['id'].endswith(f':{call+1}'))
            following_messages = following['messages']
            replayed = {'messages': following_messages[:-2], 'target': following_messages[-2]['content']}
            provenance = following['provenance']
            require(identity_digest(replayed) == step['request_sha256']
                    and provenance['trajectory_sha256'] == sha256(trace_path),
                    'The restored action must use the exact audited model context')
            result.append(dict(replayed, id=f'053:{episode}:{call}', episode_id=episode,
                               record_group_kind=step['kind'], provenance=provenance,
                               encoding_sha256=step['encoding_sha256']))
    return result


def dataset(root, design, previous):
    initial = root / previous['data_path']
    pilot_export = root / PILOT / 'private/data.jsonl'
    status_path = root / PILOT / 'status.json'
    status = read(status_path)
    require(sha256(initial) == design['original_export_sha256']
            and sha256(pilot_export) == status['export_sha256'] == design['preflight_export_sha256']
            and sha256(status_path) == design['preflight_status_sha256']
            and sha256(root/PILOT_PROTOCOL) == status['protocol_sha256'] == design['preflight_protocol_sha256']
            and status['complete'] and not status['final_test_opened']
            and status['optimizer_updates'] == status['model_generations'] == 0,
            'Require the completed, restored training-only inputs')
    original = [json.loads(line) for line in initial.read_bytes().splitlines()]
    seed = seed_targets(root, original)
    require(hash_bytes(json_lines(seed)).hexdigest() == design['seed_export_sha256'],
            'The ten restored regression decisions differ from the audited replay')
    previous_episodes = {row['episode_id']: row for row in original if row['record_group_kind'] == 'final_verification'}
    additional = [json.loads(line) for line in pilot_export.read_bytes().splitlines()]
    for row in additional:
        episode = '052:' + row['episode_id'].rsplit(':', 1)[1]
        prior = previous_episodes[episode]['provenance']
        provenance = row['provenance']
        require(all(provenance[key] == prior[key] for key in ('split','source_id','family','executed','model_generated'))
                and provenance['original_training_trace_sha256'] == prior['trajectory_sha256'],
                'Added actions must use the same reviewed source and task')
        row['episode_id'] = episode
    for row in (*seed, *additional):
        if row['record_group_kind'] == 'partial_repair_regression':
            require(parse_action(row['target']) == parse_action(previous_episodes[row['episode_id']]['target']),
                    'A partial regression must run the task’s declared check')
    combined = [*original, *seed, *additional]
    payload = json_lines(combined)
    require(hash_bytes(payload).hexdigest() == design['combined_data_sha256']
            and len({row['id'] for row in combined}) == len(combined)
            and len(previous_episodes) == design['episodes'],
            'The combined actions must match the independently restored training input')
    return combined, payload


def freeze(root):
    from transformers import AutoTokenizer

    design_path = root / EXP / 'design.json'
    design = read(design_path)
    previous = read(root / PREVIOUS)
    require(sha256(root/PREVIOUS) == design['prior_training_protocol_sha256']
            and not previous['final_test_opened'] and not previous['weights_public']
            and trainer.unchanged(root, previous['frozen_files']), 'Preserve the 052 training lineage and recipe')
    records, payload = dataset(root, design, previous)
    counts = Counter(record['provenance']['source_id'] for record in records)
    sources = read(root/previous['sources_path'])['sources']
    require(len(sources) == len(counts) == design['episodes']
            and set(counts) == {row['source_id'] for row in sources}
            and len({row['family'] for row in sources}) == design['training_families']
            and all(row['notices'] and type(row['allow_weight_publication']) is bool for row in sources),
            'The candidate must use the previously reviewed source revisions')
    sources = [dict(row, records=counts[row['source_id']]) for row in sources]
    config = {key: previous[key] for key in ('cycle_spec','runtime_path','profile_path','environment_path','model_manifest_path')}
    runtime = read(root/config['runtime_path'])
    profile = read(root/config['profile_path'])
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'], local_files_only=True, trust_remote_code=False)
    ordered = sorted(records, key=lambda row: hash_bytes(('051-repair-closure-v1\0' + row['id']).encode()).hexdigest())
    schedule = []
    for row in ordered:
        encoded = encode_training_example(row, tokenizer, profile)
        require(identity_digest(encoded['labels']) == row['encoding_sha256']
                and encoded['supervised_tokens'] <= 512, 'The original assistant labels and budget must match')
        schedule.append({'id': row['id'], 'token_count': len(encoded['ids']),
                         'supervised_token_count': encoded['supervised_tokens'], 'labels_sha256': row['encoding_sha256']})
    window = previous['recipe']['gradient_accumulation_records']
    selection = {'records': len(records), 'episodes': design['episodes'],
                 'input_tokens': sum(row['token_count'] for row in schedule),
                 'supervised_tokens': sum(row['supervised_token_count'] for row in schedule),
                 'target_counts': dict(Counter(row['record_group_kind'] for row in records)),
                 'optimizer_updates': (len(records)+window-1)//window}
    require(all(selection[key] == design[key] for key in ('target_counts','supervised_tokens','optimizer_updates')),
            'The GPU recipe must consume the exact approved training data')
    config.update(data_path=str(JOB/'data.private.jsonl'), sources_path=str(JOB/'sources.private.json'),
                  schedule_path=str(JOB/'schedule.json'))
    (root/JOB).mkdir(mode=0o700, parents=True, exist_ok=True)
    put(root/config['data_path'], payload)
    put(root/config['sources_path'], (json.dumps({'sources': sources, 'final_test_opened': False})+'\n').encode())
    put(root/config['schedule_path'], (json.dumps(schedule)+'\n').encode())
    paths = {str(EXP/'train.py'),str(EXP/'design.json'),str(PREVIOUS),str(PILOT_PROTOCOL),str(PILOT/'status.json'),
             str(PILOT/'private/data.jsonl'),str(MATERIALIZER),*config.values()}
    frozen = {**previous['frozen_files'], **{name:sha256(root/name) for name in sorted(paths)}}
    plan = {'schema_version': 1, 'candidate': EXP.name, 'kind': 'post_edit_recheck_lora',
            'model_id': previous['model_id'], 'revision': previous['revision'], 'run_directory': str(OUT),
            'expected_trainable_parameters': previous['expected_trainable_parameters'],
            'expected_records': len(records), 'recipe': previous['recipe'], 'selection': selection,
            'order_sha256': identity_digest([row['id'] for row in schedule]), **config,
            'frozen_files': frozen, 'final_test_opened': False, 'weights_public': False}
    require(runpy.run_path(str(root/MATERIALIZER))['materialize'](root,plan)[-1] == selection,
            'Independent training input materialization must reproduce the frozen schedule')
    protocol = root/JOB/'protocol.json'
    put(protocol, (json.dumps(plan)+'\n').encode())
    return {'protocol_sha256': sha256(protocol), 'design_sha256': sha256(design_path),
            'data_sha256': sha256(root/config['data_path']), 'source_review_sha256': sha256(root/config['sources_path']),
            'schedule_sha256': sha256(root/config['schedule_path']), 'order_sha256': plan['order_sha256'],
            'selection': selection, 'optimizer_updates': 0, 'final_test_opened': False}


def execute(root, phase):
    path = root/JOB/'protocol.json'
    plan = read(path)
    checksum = sha256(path)
    published = read(root/JOB/'publication.json')
    require(published['protocol_sha256'] == checksum and published['anonymous_files_verified']
            and published['ci_conclusion'] == 'success' and trainer.unchanged(root,plan['frozen_files']),
            'Require the published and unchanged training protocol')
    inputs = runpy.run_path(str(root/MATERIALIZER))['materialize'](root,plan)
    if phase == 'train':
        return trainer.train(root,plan,checksum,*inputs)
    result = trainer.audit(root,plan,checksum,*inputs[1:])
    run = root/OUT
    audited_training(run/'candidate-input-review.json',path,root/plan['profile_path'],run/'integrity.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze','train','audit'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = freeze(root) if args.phase == 'freeze' else execute(root,args.phase)
    print(json.dumps(result, allow_nan=False), flush=True)
