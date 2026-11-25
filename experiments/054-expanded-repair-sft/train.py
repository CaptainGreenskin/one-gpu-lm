"""Train one base-initialized LoRA on the expanded, verified repair data."""

import argparse
import json
import os
from pathlib import Path
import runpy

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json
from one_gpu_lm.models.release_lineage import audited_training
from one_gpu_lm.verified import continuation_training as trainer


EXP = Path('experiments/054-expanded-repair-sft')
JOB = Path('private/054-expanded-repair-sft-training')
RUN = Path('runs/q054-expanded-repair-sft')
PARENT = Path('private/053-post-edit-recheck-training/protocol.json')
MATERIALIZER = Path('experiments/051-repair-closure-sft/train.py')
INPUTS = Path('private/grounded-recheck-expanded-training-inputs')
INPUT_FILES = {
    'data_path': INPUTS / 'data.private.jsonl',
    'sources_path': INPUTS / 'sources.private.json',
    'schedule_path': INPUTS / 'schedule.private.json',
}


def read(path):
    return json.loads(path.read_bytes())


def require(valid, detail):
    if not valid:
        raise ValueError(detail)


def freeze(root):
    design_path = root / EXP / 'design.json'
    design, parent = read(design_path), read(root / PARENT)
    require(sha256(root / PARENT) == design['previous_training_protocol_sha256']
            and design['experiment'] == EXP.name
            and design['final_test_opened'] is parent['final_test_opened'] is False
            and design['weights_public'] is parent['weights_public'] is False
            and trainer.unchanged(root, parent['frozen_files']),
            'Require the frozen 053 training recipe and original inputs')
    require(all(sha256(root / path) == design['input_sha256'][key] for key, path in INPUT_FILES.items())
            and sha256(root / MATERIALIZER) == design['materializer_sha256'],
            'Require the exact independently verified expanded training material')
    settings = {key: parent[key] for key in (
        'cycle_spec', 'runtime_path', 'profile_path', 'environment_path', 'model_manifest_path',
        'model_id', 'revision', 'recipe', 'expected_trainable_parameters')}
    require((settings['model_id'], settings['revision']) == (design['model_id'], design['revision']),
            'Use the same pinned base for the expanded curriculum')
    settings.update({key: str(path) for key, path in INPUT_FILES.items()})
    files = (EXP / 'train.py', EXP / 'design.json', PARENT, MATERIALIZER, *INPUT_FILES.values())
    plan = {
        'schema_version': 1, 'candidate': EXP.name, 'kind': 'expanded_repair_closure_lora',
        'run_directory': str(RUN), 'expected_records': design['selection']['records'],
        'order_sha256': design['sample_order_sha256'], 'selection': design['selection'], **settings,
        'frozen_files': {**parent['frozen_files'], **{str(path): sha256(root / path) for path in files}},
        'final_test_opened': False, 'weights_public': False,
    }
    inputs = runpy.run_path(str(root / MATERIALIZER))['materialize'](root, plan)
    require(inputs[-1] == design['selection'], 'The verified training schedule changed')
    protocol = root / JOB / 'protocol.json'
    if protocol.is_file():
        require(not protocol.is_symlink() and read(protocol) == plan, 'Preserve the original training protocol')
    else:
        atomic_json(protocol, plan)
    return {
        'protocol_sha256': sha256(protocol), 'design_sha256': sha256(design_path),
        'data_sha256': design['input_sha256']['data_path'],
        'source_review_sha256': design['input_sha256']['sources_path'],
        'schedule_sha256': design['input_sha256']['schedule_path'],
        'order_sha256': plan['order_sha256'], 'selection': plan['selection'],
        'optimizer_updates': 0, 'final_test_opened': False,
    }


def execute(root, phase):
    protocol = root / JOB / 'protocol.json'
    plan = read(protocol)
    checksum = sha256(protocol)
    public = read(root / JOB / 'publication.json')
    require(public['protocol_sha256'] == checksum and public['anonymous_files_verified']
            and public['ci_conclusion'] == 'success' and trainer.unchanged(root, plan['frozen_files']),
            'Train and audit only the published frozen protocol')
    inputs = runpy.run_path(str(root / MATERIALIZER))['materialize'](root, plan)
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
