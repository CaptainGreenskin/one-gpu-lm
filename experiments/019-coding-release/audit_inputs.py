from pathlib import Path
import hashlib,json,re,sys
needles=json.loads(sys.stdin.read());root=Path('.').resolve();exp=Path('experiments/019-coding-release')
data=Path('runs/prepared-data/train.jsonl');rows=[json.loads(line) for line in data.read_text().splitlines()]
patterns={'github_token':r'gh[pousr]_[A-Za-z0-9]{30,}','github_fine_grained':r'github_pat_[A-Za-z0-9_]{40,}','private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----','aws_access_key':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'}
findings=[]
for row in rows:
 text=json.dumps(row,ensure_ascii=False)
 cats=[name for name,p in patterns.items() if re.search(p,text)]
 if any(n and n in text for n in needles):cats.append('known_private_identifier')
 if cats:findings.append({'record_id_sha256':hashlib.sha256(row['id'].encode()).hexdigest(),'categories':cats})
ledger=json.loads((exp/'sources.json').read_text());notices={}
for source in ledger['sources']:
 if source['split']!='train':continue
 p=exp/'notices'/(source['repository'].replace('/','--')+'-LICENSE.txt')
 digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==source['license_sha256']
 notices[source['repository']]=digest
reference=json.loads(Path('runs/reference-data/receipt.json').read_text());assert reference['teacher_api_used'] is False
result={'schema_version':1,'passed':not findings,'records_checked':len(rows),'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'known_secret_pattern_findings':findings,'known_private_identifiers_present':any('known_private_identifier' in f['categories'] for f in findings),'exact_training_source_notices':notices,'teacher_api_used':False,'scope':'Known credential patterns and supplied private endpoint/account/run identifiers in actual training JSONL; exact source notices retained. Does not assert exhaustive secret detection or settle all legal questions.'}
Path('training-input-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if findings:raise SystemExit(1)
