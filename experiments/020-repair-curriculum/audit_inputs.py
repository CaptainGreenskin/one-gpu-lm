"""Bounded private-identifier and credential-pattern screen of both actual datasets."""
from pathlib import Path
import hashlib
import json
import re
import sys

root = Path('.').resolve(); exp = Path('experiments/020-repair-curriculum')
needles = json.loads(sys.stdin.read())
patterns = {'github_token':r'gh[pousr]_[A-Za-z0-9]{30,}', 'github_fine_grained':r'github_pat_[A-Za-z0-9_]{40,}',
    'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', 'aws_access_key':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'}
findings = []; hashes = {}; counts = {}
for variant in ['repair','recovery']:
    data = Path('runs/prepared-data020')/variant/'train.jsonl'
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    hashes[variant] = hashlib.sha256(data.read_bytes()).hexdigest(); counts[variant] = len(rows)
    for row in rows:
        text = json.dumps(row,ensure_ascii=False)
        categories = [name for name,p in patterns.items() if re.search(p,text)]
        if any(n and n in text for n in needles): categories.append('known_private_identifier')
        if categories: findings.append({'variant':variant,'record_id_sha256':hashlib.sha256(row['id'].encode()).hexdigest(),'categories':categories})
    reference = json.loads((Path('runs/reference-data020')/variant/'receipt.json').read_text())
    assert reference['teacher_api_used'] is False and reference['development_examples_used'] is False
notices = {}
for source in json.loads((exp/'sources.json').read_text())['sources']:
    if source['split'] != 'train': continue
    path = exp/'notices'/(source['repository'].replace('/','--')+'-LICENSE.txt')
    digest = hashlib.sha256(path.read_bytes()).hexdigest(); assert digest == source['license_sha256']
    notices[source['repository']] = digest
value = {'schema_version':1,'passed':not findings,'records_checked':counts,'data_sha256':hashes,
    'known_secret_pattern_findings':findings,'exact_training_source_notices':notices,
    'teacher_api_used':False,'development_examples_used':False,
    'scope':'Actual paired training JSONL, known credential patterns and supplied private identifiers; exact upstream notices. Not exhaustive secret detection or a legal warranty.'}
out = Path('training-input-audit.json'); assert not out.exists(); out.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'passed':value['passed'],'records_checked':counts,'findings':len(findings)}))
if findings: raise SystemExit(1)
