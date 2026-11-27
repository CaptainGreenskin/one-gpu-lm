"""Train one base-initialized LoRA with the audited multi-file repair additions."""

import argparse
import json
import os
from pathlib import Path
import runpy

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json
from one_gpu_lm.models.release_lineage import audited_training
from one_gpu_lm.verified import continuation_training as trainer


EXP = Path('experiments/057-multifile-correction-sft')
JOB = Path('private/057-multifile-correction-sft-training')
RUN = Path('runs/q057-multifile-correction-sft')
PARENT = Path('private/054-expanded-repair-sft-training/protocol.json')
MATERIALIZER = Path('experiments/051-repair-closure-sft/train.py')
INPUTS = Path('private/057-multifile-correction-training-inputs')
FILES = {
    'data_path': INPUTS / 'data.private.jsonl',
    'sources_path': INPUTS / 'sources.private.json',
    'schedule_path': INPUTS / 'schedule.private.json',
}


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def materialize(root, plan):
    return runpy.run_path(str(root / MATERIALIZER))['materialize'](root, plan)


def freeze(root):
    design_path = root / EXP / 'design.json'
    design, parent = read(design_path), read(root / PARENT)
    require(sha256(root / PARENT) == design['parent_training_protocol_sha256']
            and parent['candidate'] == '054-expanded-repair-sft'
            and design['experiment'] == EXP.name
            and (parent['model_id'], parent['revision']) == (design['model_id'], design['revision'])
            and parent['recipe'] == design['recipe']
            and all(document['final_test_opened'] is document['weights_public'] is False
                    for document in (design, parent))
            and trainer.unchanged(root, parent['frozen_files']),
            'Require the audited parent training settings and original inputs')
    require(sha256(root / MATERIALIZER) == design['materializer_sha256']
            == parent['frozen_files'][str(MATERIALIZER)]
            and all(sha256(root / path) == design['input_sha256'][key] for key, path in FILES.items()),
            'Use the exact audited training mixture and unchanged materializer')
    added = design['additions']
    before = parent['selection']
    after = design['selection']
    require(added['actions'] > 0 and added['verified_multifile_repairs'] > 0
            and after['records'] == before['records'] + added['actions']
            and after['episodes'] == before['episodes'] + added['verified_multifile_repairs']
            and (root / FILES['data_path']).read_bytes().startswith((root / parent['data_path']).read_bytes()),
            'Require every original record and the expected new repair episodes')
    paths = (EXP / 'train.py', EXP / 'design.json', PARENT, *FILES.values())
    plan = dict(parent, candidate=EXP.name, kind='multifile_correction_lora', run_directory=str(RUN),
                expected_records=after['records'], order_sha256=design['sample_order_sha256'], selection=after,
                frozen_files={**parent['frozen_files'], **{str(path): sha256(root / path) for path in paths}},
                **{key: str(path) for key, path in FILES.items()})
    *_, selection = materialize(root, plan)
    require(selection == after, 'Verify the exact scheduled actions and assistant labels before GPU access')
    protocol = root / JOB / 'protocol.json'
    require(not protocol.is_symlink(), 'Use an ordinary protocol file')
    if protocol.exists():
        require(read(protocol) == plan, 'Preserve the frozen training protocol')
    else:
        atomic_json(protocol, plan)
    return {
        'protocol_sha256': sha256(protocol), 'design_sha256': sha256(design_path),
        'data_sha256': design['input_sha256']['data_path'],
        'source_review_sha256': design['input_sha256']['sources_path'],
        'schedule_sha256': design['input_sha256']['schedule_path'],
        'order_sha256': plan['order_sha256'], 'selection': selection, 'optimizer_updates': 0,
        'final_test_opened': False,
    }


def execute(root, phase):
    protocol = root / JOB / 'protocol.json'
    plan, checksum = read(protocol), sha256(protocol)
    publication = read(root / JOB / 'publication.json')
    require(publication['protocol_sha256'] == checksum and publication['anonymous_files_verified'] is True
            and publication['ci_conclusion'] == 'success' and trainer.unchanged(root, plan['frozen_files']),
            'Use the published and unchanged training protocol')
    inputs = materialize(root, plan)
    if phase == 'train':
        return trainer.train(root, plan, checksum, *inputs)
    result = trainer.audit(root, plan, checksum, *inputs[1:])
    out = root / RUN
    audited_training(out / 'candidate-input-review.json', protocol, root / plan['profile_path'], out / 'integrity.json')
    return result


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'train', 'audit'))
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = freeze(root) if args.phase == 'freeze' else execute(root, args.phase)
    print(json.dumps(result, allow_nan=False), flush=True)
