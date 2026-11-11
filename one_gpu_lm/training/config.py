"""Strict local run configuration; paths stay private, input identities are publishable."""
import json
import os
from pathlib import Path

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.training_data import read_training_data


INPUTS = {'model_dir','data','data_receipt','admission','licenses'}
RUN_FIELDS = {'max_steps','max_examples','checkpoint_every','minimum_free_gpu_gib',
              'minimum_free_disk_gib','max_attempt_seconds','telemetry_interval_seconds'}


def resolve_path(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError('Expected a nonempty local path')
    expanded = os.path.expandvars(value)
    if '$' in expanded:
        raise ValueError('Set every environment variable used by the run configuration')
    path = Path(expanded).expanduser()
    return (base/path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if set(value) != {'schema_version','profile','weights_manifest','environment_lock','inputs','run'} or value['schema_version'] != 1:
        raise ValueError('Unsupported or misspelled run configuration fields')
    if set(value['inputs']) != INPUTS or set(value['run']) != RUN_FIELDS:
        raise ValueError('Unsupported or missing input/budget fields')
    for key in RUN_FIELDS:
        number = value['run'][key]
        if isinstance(number, bool) or not isinstance(number, (int,float)) or not 0 < number < float('inf'):
            raise ValueError('Run budgets must be finite and positive: '+key)
    for key in ['max_steps','max_examples','checkpoint_every']:
        if not isinstance(value['run'][key], int):
            raise ValueError('Expected an integer budget: '+key)
    paths = {key:resolve_path(value[key], path.parent) for key in ['profile','weights_manifest','environment_lock']}
    paths.update({key:resolve_path(p, path.parent) for key,p in value['inputs'].items()})
    if not paths['model_dir'].is_dir() or any(not p.is_file() for key,p in paths.items() if key!='model_dir'):
        raise ValueError('A configured model directory or input file is missing')
    profile = json.loads(paths['profile'].read_text())
    if profile['training'].get('dialogue_format', 'legacy') not in {'legacy', 'native_roundtrip'}:
        raise ValueError('Unsupported training dialogue format')
    if profile['dtype'] != 'bfloat16' or profile['training']['epochs'] != 1 or not profile['training']['gradient_checkpointing']:
        raise ValueError('Infrastructure v1 supports BF16, one data pass, and gradient checkpointing')
    if profile['training']['max_sequence_length'] > 4096 or profile['lora']['r'] > 8:
        raise ValueError('This validated recipe supports at most 4096 tokens and rank 8')
    if not 0 < profile['training']['learning_rate'] < 1 or profile['training']['max_grad_norm'] <= 0:
        raise ValueError('Invalid optimizer configuration')
    return {'paths':paths,'profile':profile,'run':value['run'],
            'input_hashes':{key:sha256(p) for key,p in paths.items() if key!='model_dir'}}


