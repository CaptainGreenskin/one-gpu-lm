"""Train the context variant with the original model, targets and sample order."""

import argparse
import json
import os
from pathlib import Path
import runpy

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json
from one_gpu_lm.models.release_lineage import audited_training
from one_gpu_lm.verified import continuation_training as trainer


EXP = Path('experiments/058-read-history-replay')
JOB = Path('private/058-read-history-training')
RUN = Path('runs/q058-read-history')
PARENT = Path('private/057-multifile-correction-sft-training/protocol.json')
INPUTS = Path('private/058-read-history-training-inputs')
MATERIALIZER = Path('experiments/051-repair-closure-sft/train.py')
FILES = dict(data_path=INPUTS/'data.private.jsonl', sources_path=INPUTS/'sources.private.json',
             schedule_path=INPUTS/'schedule.private.json')


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def materialize(root, plan):
    return runpy.run_path(str(root / MATERIALIZER))['materialize'](root, plan)


def check_records(original, revised, expected):
    before = [json.loads(line) for line in original.read_bytes().splitlines()]
    after = [json.loads(line) for line in revised.read_bytes().splitlines()]
    changed = 0
    for old, new in zip(before, after, strict=True):
        if old == new:
            continue
        evidence = new['provenance']['read_history_receipt_sha256']
        messages = new['messages']
        require(len(evidence) == 64
                and new['provenance'] == dict(old['provenance'], read_history_receipt_sha256=evidence)
                and {key: new[key] for key in ('id', 'episode_id', 'target', 'record_group_kind')}
                == {key: old[key] for key in ('id', 'episode_id', 'target', 'record_group_kind')}
                and messages[:2] == old['messages'][:2] and messages[6:] == old['messages'][2:]
                and [message['role'] for message in messages[2:6]] == ['assistant', 'user'] * 2,
                'Each context variant must keep its original question, target, role and preceding actions')
        changed += 1
    require(changed == expected, 'The revised record count changed')


def freeze(root):
    design_path = root / EXP / 'design.json'
    design, parent = read(design_path), read(root / PARENT)
    status = INPUTS / 'build-status.json'
    evidence = read(root / status)
    require(sha256(root / PARENT) == design['parent_training_protocol_sha256']
            and sha256(root / status) == design['materialization_status_sha256']
            and parent['candidate'] == '057-multifile-correction-sft' and evidence['complete']
            and not parent['weights_public'] and not any(row['final_test_opened'] for row in (evidence, design, parent))
            and evidence['revised_records'] == design['revised_records'] > 0
            and evidence['previous_selection'] == parent['selection']
            and trainer.unchanged(root, parent['frozen_files']),
            'Use the original audited training plan and completed context materialization')
    selection = design['selection']
    require(selection == evidence['selection']
            and {key: val for key, val in selection.items() if key != 'input_tokens'}
            == {key: val for key, val in parent['selection'].items() if key != 'input_tokens'}
            and design['sample_order_sha256'] == parent['order_sha256']
            and design['recipe'] == parent['recipe']
            and all(sha256(root / path) == design['input_sha256'][key] for key, path in FILES.items())
            and design['input_sha256']['sources_path'] == sha256(root / parent['sources_path']),
            'Preserve the original targets, source inventory, optimizer and order')
    check_records(root / parent['data_path'], root / FILES['data_path'], design['revised_records'])
    names = (EXP/'train.py', EXP/'design.json', EXP/'plan.py', PARENT, status, *FILES.values())
    plan = dict(parent, candidate=EXP.name, kind='read_history_lora', run_directory=str(RUN), selection=selection,
                frozen_files={**parent['frozen_files'], **{str(path): sha256(root/path) for path in names}},
                **{key: str(path) for key, path in FILES.items()})
    *_, measured = materialize(root, plan)
    require(measured == selection, 'Rerun the tokenizer and label checks before assigning GPU time')
    path = root / JOB / 'protocol.json'
    if path.exists():
        require(read(path) == plan, 'Preserve the frozen 058 training plan')
    else:
        atomic_json(path, plan)
    return dict(protocol_sha256=sha256(path), design_sha256=sha256(design_path),
                data_sha256=design['input_sha256']['data_path'], source_review_sha256=design['input_sha256']['sources_path'],
                schedule_sha256=design['input_sha256']['schedule_path'], order_sha256=plan['order_sha256'],
                selection=selection, optimizer_updates=0, final_test_opened=False)


def execute(root, phase):
    path = root / JOB / 'protocol.json'
    plan, checksum = read(path), sha256(path)
    approval = read(root / JOB / 'publication.json')
    require(approval['protocol_sha256'] == checksum and approval['anonymous_files_verified']
            and approval['ci_conclusion'] == 'success' and trainer.unchanged(root, plan['frozen_files']),
            'Use the same trained inputs as the verified public registration')
    inputs = materialize(root, plan)
    if phase == 'train':
        return trainer.train(root, plan, checksum, *inputs)
    evidence = trainer.audit(root, plan, checksum, *inputs[1:])
    output = root / RUN
    audited_training(output/'candidate-input-review.json', path, root/plan['profile_path'], output/'integrity.json')
    return evidence


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'train', 'audit'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    print(json.dumps(freeze(root) if args.phase == 'freeze' else execute(root, args.phase), allow_nan=False))
