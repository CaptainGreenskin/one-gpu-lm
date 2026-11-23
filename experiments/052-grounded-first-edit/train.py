"""Freeze and train the grounded first-edit LoRA with the established 051 recipe."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
import runpy
import sys

from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.privacy import verify_private_labels
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, identity_digest
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.models.release_lineage import audited_training
from one_gpu_lm.verified import continuation_training as trainer


EXP = Path('experiments/052-grounded-first-edit')
COLLECTION = Path('private/052-grounded-first-edit')
DATA = Path('runs/q052-grounded-first-edit')
JOB = Path('private/052-grounded-first-edit-training')
OUT = Path('runs/q052-grounded-first-edit-sft')
PREVIOUS_RECIPE = Path('private/051-repair-closure-sft/protocol.json')
PREVIOUS_TRAINER = Path('experiments/051-repair-closure-sft/train.py')


def read(path):
    return json.loads(path.read_bytes())


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def freeze_object(path, value):
    if path.is_file():
        require(read(path) == value, 'Preserve the first registered training object')
    else:
        atomic_json(path, value)


def review_collection(root, collection, observed):
    sys.path.insert(0, str(root/collection['closure_replay_code']))
    import run as original
    collector = runpy.run_path(str(root/COLLECTION/'collect.py'))
    cases = collector['selected_cases'](root, collection, original)
    rows = observed['rows']
    require(len(cases) == len(rows) == collection['design']['candidate_episodes']
            and [row['case'] for row in rows] == list(range(1,len(rows)+1)),
            'The terminal collection does not match the registered population')
    groups = defaultdict(list)
    export = root/DATA/'private/data.jsonl'
    data = [json.loads(line) for line in export.read_bytes().splitlines()]
    require(len(data) == observed['export_rows'] and sha256(export) == observed['export_sha256'],
            'Use the exact independently executed collection export')
    require(not any(verify_private_labels(data, ()).values()), 'The training export contains sensitive patterns')
    for record in data:
        groups[record['episode_id']].append(record)
    sources = []
    for row, (case, _, _, choice) in zip(rows,cases):
        key = f"052:{row['case']:03d}"
        selected = groups.pop(key, [])
        require(bool(selected) == row['qualified'], 'Unqualified repair entered the training export')
        if not selected:
            continue
        source = case.source
        folder = root/DATA/'private'/f"{row['case']:03d}"
        trace_path = folder/'actor-trace.json'
        trace = read(trace_path)
        grade = read(folder/'grade.json')
        grader = read(folder/'grader-output.json')
        require(read(folder/'export.json') == selected and grade['passed']
                and grader['returncode'] == grade['returncode'] == 0
                and hash_bytes(grader['output'].encode()).hexdigest() == grade['grader_output_sha256']
                and grade['source_sha256'] == {name:source['files'][name] for name in case.task['editable_files']}
                and all(record['provenance']['trajectory_sha256'] == sha256(trace_path)
                        and record['provenance']['source_id'] == source['source_id'] for record in selected),
                'The protected grade or executed trajectory differs from its training targets')
        first = selected[0]
        edit = parse_action(first['target'])['command']
        _, evidence = collector['inspection'](case.task,{'command':edit,
                                    'source_file':next(iter(trace['steps'][1]['changed']))})
        kinds = [item['record_group_kind'] for item in selected]
        require(hash_bytes(edit.encode()).hexdigest() == choice['first_edit_sha256']
                and kinds[-2:] == ['final_verification','explicit_submission']
                and set(kinds[:-2]) == {'reference_edit'} and len(first['messages']) == 4
                and len(evidence) == row['source_evidence_fragments']
                and all(piece in first['messages'][3]['content'] for piece in evidence),
                'The first edit lacks source evidence in the actual tool feedback')
        sources.append({'source_id':source['source_id'],'repository':source['repository'],
                        'family':source['family'],'revision':source['revision'],'parent':source['upstream_parent'],
                        'license':source['license'],'records':len(selected),
                        'notices':{name:source['files'][name] for name in source['notices']},
                        'allow_weight_publication':source['approved_for']['weight_publication'],
                        'aliases':source.get('related_families', [])})
    require(not groups, 'Every exported record must have one independently graded training episode')
    qualified = [row for row in rows if row['qualified']]
    multi = [row for row in qualified if row['multifile']]
    coverage = {'episodes':len(qualified),'multifile_episodes':len(multi),
                'families':len({row['family'] for row in qualified}),
                'multifile_families':len({row['family'] for row in multi})}
    require(len(sources) == coverage['episodes'] and all(
        count >= collection['design']['data_gate']['minimum_'+key] for key,count in coverage.items()),
        'The audited grounded examples must meet the frozen source coverage gate')
    return data, sources, coverage


def freeze(root):
    from transformers import AutoTokenizer

    collection_path = root/COLLECTION/'protocol.json'
    collection = read(collection_path)
    registered = read(root/EXP/'collection-registration.json')
    status_path = root/DATA/'status.json'
    status = read(status_path)
    require(registered['collection_protocol_sha256'] == sha256(collection_path) == status['protocol_sha256']
            and registered['collector_code_sha256'] == sha256(root/COLLECTION/'collect.py')
            and status['complete'] and status['data_gate_passed'] and not status['final_test_opened']
            and status['model_generations'] == status['optimizer_updates'] == 0,
            'Require the completed registered training-only source replay')
    previous = read(root/PREVIOUS_RECIPE)
    previous_registration = Path('experiments/051-repair-closure-sft/registration.json')
    previous_receipt = read(root/previous_registration)
    require(sha256(root/PREVIOUS_RECIPE) == previous_receipt['protocol_sha256']
            and previous_receipt['source_sha256'][str(PREVIOUS_TRAINER)] == sha256(root/PREVIOUS_TRAINER)
            and (previous['model_id'],previous['revision']) == (registered['model_id'],registered['revision']),
            'Preserve the audited 051 base and optimizer recipe')
    records, sources, coverage = review_collection(root, collection, status)
    profile = read(root/previous['profile_path'])
    runtime = read(root/previous['runtime_path'])
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False)
    order = sorted(records,key=lambda row: hash_bytes(('051-repair-closure-v1\0'+row['id']).encode()).hexdigest())
    schedule = []
    for row in order:
        encoded = encode_training_example(row,tokenizer,profile)
        require(identity_digest(encoded['labels']) == row['encoding_sha256']
                and encoded['supervised_tokens'] <= collection['design']['evaluation_budget']['max_generated_tokens_per_call'],
                'A supervised target differs from the audited CPU replay')
        schedule.append({'id':row['id'],'token_count':len(encoded['ids']),
                         'supervised_token_count':encoded['supervised_tokens'],'labels_sha256':row['encoding_sha256']})
    window = previous['recipe']['gradient_accumulation_records']
    summary = {'records':len(order),'episodes':coverage['episodes'],
               'input_tokens':sum(row['token_count'] for row in schedule),
               'supervised_tokens':sum(row['supervised_token_count'] for row in schedule),
               'target_counts':dict(Counter(row['record_group_kind'] for row in order)),
               'optimizer_updates':(len(order)+window-1)//window}
    phase = root/JOB
    phase.mkdir(mode=0o700,parents=True,exist_ok=True)
    source_path, schedule_path = JOB/'sources.private.json', JOB/'schedule.json'
    freeze_object(root/source_path,{'sources':sources,'final_test_opened':False})
    freeze_object(root/schedule_path,schedule)
    cfg = {key:previous[key] for key in ('cycle_spec','runtime_path','profile_path','environment_path','model_manifest_path')}
    cfg.update(data_path=str(DATA/'private/data.jsonl'),sources_path=str(source_path),schedule_path=str(schedule_path))
    pinned = [*(root/'one_gpu_lm').rglob('*.py'),*(root/path for path in cfg.values()),
              root/previous_registration,root/PREVIOUS_RECIPE,root/PREVIOUS_TRAINER,root/EXP/'train.py',
              root/EXP/'collection-registration.json',collection_path,status_path,root/COLLECTION/'collect.py']
    plan = {'schema_version':1,'candidate':collection['design']['experiment'],'kind':'grounded_first_edit_lora',
            'model_id':previous['model_id'],'revision':previous['revision'],'run_directory':str(OUT),
            'expected_trainable_parameters':previous['expected_trainable_parameters'],'expected_records':len(order),
            'recipe':previous['recipe'],'selection':summary,'coverage':coverage,
            'order_sha256':identity_digest([row['id'] for row in schedule]),**cfg,
            'final_test_opened':False,'weights_public':False,
            'frozen_files':{str(path.relative_to(root)):sha256(path) for path in sorted(set(pinned))}}
    materialize = runpy.run_path(str(root/PREVIOUS_TRAINER))['materialize']
    require(materialize(root,plan)[-1] == summary, 'Training and independent encoding disagree')
    freeze_object(phase/'protocol.json',plan)
    return {'protocol_sha256':sha256(phase/'protocol.json'),'design_sha256':sha256(root/EXP/'design.json'),
            'source_review_sha256':sha256(root/source_path),'schedule_sha256':sha256(root/schedule_path),
            'order_sha256':plan['order_sha256'],'selection':summary,'coverage':coverage,
            'collection_status_sha256':sha256(status_path),'optimizer_updates':0,'final_test_opened':False}


def execute(root, phase):
    path = root/JOB/'protocol.json'
    plan = read(path)
    checksum = sha256(path)
    published = read(root/JOB/'publication.json')
    require(published['protocol_sha256'] == checksum and published['anonymous_files_verified']
            and published['ci_conclusion'] == 'success' and trainer.unchanged(root,plan['frozen_files']),
            'Require the publicly registered training run with unchanged inputs')
    inputs = runpy.run_path(str(root/PREVIOUS_TRAINER))['materialize'](root,plan)
    if phase == 'train':
        return trainer.train(root,plan,checksum,*inputs)
    result = trainer.audit(root,plan,checksum,*inputs[1:])
    run = root/OUT
    audited_training(run/'candidate-input-review.json',path,root/plan['profile_path'],run/'integrity.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','train','audit'))
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = freeze(root) if args.phase == 'freeze' else execute(root,args.phase)
    print(json.dumps(result,allow_nan=False),flush=True)
