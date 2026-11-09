"""Shared training-source, partition and record admission contract."""
import json
from .receipts import sha256


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
