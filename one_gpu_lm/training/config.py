"""Strict local run configuration; paths stay private, input identities are publishable."""
import json
import os
from pathlib import Path

from one_gpu_lm.common.receipts import sha256


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
    if profile['dtype'] != 'bfloat16' or profile['training']['epochs'] != 1 or not profile['training']['gradient_checkpointing']:
        raise ValueError('Infrastructure v1 supports BF16, one data pass, and gradient checkpointing')
    if profile['training']['max_sequence_length'] > 4096 or profile['lora']['r'] > 8:
        raise ValueError('This validated recipe supports at most 4096 tokens and rank 8')
    if not 0 < profile['training']['learning_rate'] < 1 or profile['training']['max_grad_norm'] <= 0:
        raise ValueError('Invalid optimizer configuration')
    return {'paths':paths,'profile':profile,'run':value['run'],
            'input_hashes':{key:sha256(p) for key,p in paths.items() if key!='model_dir'}}


def read_training_data(config):
    paths = config['paths']
    receipt = json.loads(paths['data_receipt'].read_text())
    admission = json.loads(paths['admission'].read_text())
    licenses = json.loads(paths['licenses'].read_text())
    if receipt['sha256'] != sha256(paths['data']) or receipt['license_record_sha256'] != sha256(paths['licenses']):
        raise ValueError('Data or source-license identity changed')
    if not admission['complete'] or receipt['task_spec_sha256'] != admission['task_spec_sha256']:
        raise ValueError('Incomplete or mismatched task admission')
    approved = {t['id']:t for t in admission['tasks'] if t['admitted'] and t['split']=='train'}
    sources = {(s['repository'],s['revision']) for s in licenses['sources']
               if s['split']=='train' and s.get('license_sha256') and s.get('rights_review',{}).get('status')}
    rows = []
    for line in paths['data'].read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        provenance = row['provenance']
        task = approved.get(provenance['task_id'])
        if provenance['split']!='train' or task is None or task['repository']!=provenance['repository']:
            raise ValueError('Evaluation or unadmitted task entered training data')
        if (provenance['repository'],provenance['revision']) not in sources:
            raise ValueError('Unregistered training source or revision')
        if provenance['kind'] not in {'verified-local-base-trajectory','verified-reference-repair'}:
            raise ValueError('Unsupported supervision provenance kind')
        if not row['target'].strip() or not row['messages'] or row['messages'][-1]['role']!='user':
            raise ValueError('Invalid assistant training example')
        if any(m['role'] not in {'system','user','assistant'} or not isinstance(m['content'],str) for m in row['messages']):
            raise ValueError('Unsupported message role/content')
        rows.append(row)
    if len(rows)!=receipt['records'] or len({r['id'] for r in rows})!=len(rows) or not rows:
        raise ValueError('Empty, duplicated or incomplete training records')
    return rows, set(approved)
