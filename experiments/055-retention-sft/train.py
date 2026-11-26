"""Train an equal-data, lower-learning-rate control for the 054 adapter."""

import argparse
import json
import os
from pathlib import Path
import runpy

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json
from one_gpu_lm.models.release_lineage import audited_training
from one_gpu_lm.verified import continuation_training as trainer


EXP = Path('experiments/055-retention-sft')
JOB = Path('private/055-retention-sft-training')
RUN = Path('runs/q055-retention-sft')
PARENT = Path('private/054-expanded-repair-sft-training/protocol.json')
MATERIALIZER = Path('experiments/051-repair-closure-sft/train.py')
INPUTS = ('data_path', 'sources_path', 'schedule_path')


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fixed(path, value):
    require(not path.is_symlink(), 'Require an ordinary training configuration file')
    if path.exists():
        require(read(path) == value, 'Preserve the original training configuration')
    else:
        atomic_json(path, value)


def materialize(root, plan):
    return runpy.run_path(str(root / MATERIALIZER))['materialize'](root, plan)


def freeze(root):
    design_path = root / EXP / 'design.json'
    design, parent = read(design_path), read(root / PARENT)
    require(sha256(root / PARENT) == design['parent_training_protocol_sha256']
            and parent['candidate'] == '054-expanded-repair-sft' and design['experiment'] == EXP.name
            and (parent['model_id'], parent['revision']) == (design['model_id'], design['revision'])
            and all(document['final_test_opened'] is document['weights_public'] is False
                    for document in (design, parent))
            and trainer.unchanged(root, parent['frozen_files']), 'Require the exact audited 054 training contract')
    require(parent['selection'] == design['selection'] and parent['order_sha256'] == design['sample_order_sha256']
            and all(sha256(root / parent[key]) == design['input_sha256'][key] for key in INPUTS),
            'Reuse the complete 054 data, source review and action order')
    require(parent['frozen_files'][str(MATERIALIZER)] == sha256(root / MATERIALIZER),
            'Use the original label encoding and source review implementation')

    baseline_rate = design['control_learning_rate']
    rate = design['candidate_learning_rate']
    spec = read(root / parent['cycle_spec'])
    profile = read(root / parent['profile_path'])
    require(parent['recipe']['learning_rate'] == spec['candidate']['learning_rate']
            == profile['training']['learning_rate'] == baseline_rate == 1e-5 and rate == baseline_rate / 2,
            'Allow only the registered half-rate AdamW control')

    (root / JOB).mkdir(mode=0o700, parents=True, exist_ok=True)
    spec_path, profile_path = JOB / 'cycle-spec.json', JOB / 'training-profile.json'
    fixed(root / spec_path, dict(spec, candidate=dict(spec['candidate'], learning_rate=rate)))
    fixed(root / profile_path, dict(profile, training=dict(profile['training'], learning_rate=rate)))
    extra = (EXP / 'train.py', EXP / 'design.json', PARENT, spec_path, profile_path)
    plan = dict(parent, candidate=EXP.name, kind='retention_learning_rate_control_lora', run_directory=str(RUN),
                cycle_spec=str(spec_path), profile_path=str(profile_path),
                recipe=dict(parent['recipe'], learning_rate=rate),
                frozen_files={**parent['frozen_files'], **{str(path): sha256(root / path) for path in extra}})
    inputs = materialize(root, plan)
    require(inputs[-1] == design['selection'] and inputs[1].recipe['learning_rate'] == rate,
            'The effect of the control is limited to the optimizer learning rate')
    protocol = root / JOB / 'protocol.json'
    fixed(protocol, plan)
    return {'protocol_sha256': sha256(protocol), 'design_sha256': sha256(design_path),
            'data_sha256': design['input_sha256']['data_path'],
            'source_review_sha256': design['input_sha256']['sources_path'],
            'schedule_sha256': design['input_sha256']['schedule_path'],
            'order_sha256': plan['order_sha256'], 'selection': plan['selection'],
            'optimizer_updates': 0, 'learning_rate': rate, 'final_test_opened': False}


def execute(root, phase):
    protocol = root / JOB / 'protocol.json'
    plan, checksum = read(protocol), sha256(protocol)
    publication = read(root / JOB / 'publication.json')
    require(publication['protocol_sha256'] == checksum and publication['anonymous_files_verified'] is True
            and publication['ci_conclusion'] == 'success' and trainer.unchanged(root, plan['frozen_files']),
            'Run only the exact publicly verified training protocol')
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
