"""Post-hoc read-only inspection of every first residual feedback difference.

This never changes model feedback, ordering, frozen rules or primary metrics.
Only hashes, counts and classifications may leave the private journal store.
"""
import argparse,json,re,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json
from _feedback024 import module
norm=module('normalize');analysis=module('analysis')
ROW=re.compile(r'^\s+[✔✓✘✖] [^\r\n]+\r?\n?$')

def classify(a,b):
 if a==b:raise ValueError('Not a residual feedback difference')
 x,y=a.splitlines(keepends=True),b.splitlines(keepends=True)
 same=Counter(x)==Counter(y)
 lo=0
 while lo<min(len(x),len(y)) and x[lo]==y[lo]:lo+=1
 hi=0
 while hi<min(len(x),len(y))-lo and x[-1-hi]==y[-1-hi]:hi+=1
 changed=x[lo:len(x)-hi if hi else len(x)]+y[lo:len(y)-hi if hi else len(y)]
 status_only=bool(changed) and all(ROW.fullmatch(line) for line in changed)
 both_truncated=a.endswith(norm.MARKER) and b.endswith(norm.MARKER)
 pa,pb=(a[:-len(norm.MARKER)],b[:-len(norm.MARKER)]) if both_truncated else (a,b)
 prefix_only=both_truncated and (pa.startswith(pb) or pb.startswith(pa))
 return {'category':'v2_truncated_prefix_length' if prefix_only else 'v2_test_row_order' if same and status_only else 'v2_other_text',
  'both_raw_prefixes_truncated':both_truncated,'shorter_v2_payload_is_exact_prefix':prefix_only,
  'v2_payload_chars':[len(pa),len(pb)],
  'v2_line_multisets_equal':same,'changed_region_only_test_status_rows':status_only,
  'v2_sha256':[norm.digest(a),norm.digest(b)],'v2_line_counts':[len(x),len(y)]}

def review(root,run):
 root,run=Path(root),Path(run);exp=root/'experiments/024-feedback-noise'
 audit=json.loads((run/'execution-audit.json').read_text())
 assert json.loads((run/'postrun-status.json').read_text())['complete']
 campaigns={p.parent.name:json.loads(p.read_text()) for p in (run/'campaigns').glob('*/campaign.json')}
 value=analysis.summarize(campaigns,audit);pairs=[p for p in value['pairs'] if p['category']=='other_tool_text']
 traces={n:{r['id']:r for r in c['tasks']} for n,c in audit['campaigns'].items()};records=[]
 for pair in pairs:
  feedback=[];delivered=[]
  for key in ['a','b']:
   cell=pair[key];trace=traces[cell][pair['id']]
   index=next(i for i,o in enumerate(trace['observation_receipts']) if o['model_call']==pair['call'])
   p=run/'campaigns'/cell/'private'/pair['id']/'observations.jsonl'
   assert sha256(p)==trace['journal_sha256']
   row=[json.loads(line) for line in p.read_text().splitlines()][index];receipt=trace['observation_receipts'][index]
   assert norm.digest(row['raw_output'])==receipt['raw_sha256'] and norm.digest(row['delivered_output'])==receipt['delivered_sha256']
   projected=norm.transform(row['raw_output'],'extended')[0]
   assert norm.digest(projected)==receipt['canonical_sha256']
   feedback.append(projected);delivered.append(receipt['delivered_sha256'])
  keys=['comparison','model','condition','a','b','id','call','outcome_disagreement','action_disagreement']
  records.append({k:pair[k] for k in keys}|classify(*feedback)|{'delivered_sha256':delivered})
 return {'schema_version':1,'post_hoc':True,'preregistered_metric':False,'normalizer_changed':False,
  'source_sha256':sha256(Path(__file__)),'analysis_sha256':sha256(exp/'analysis.py'),
  'execution_audit_sha256':sha256(run/'execution-audit.json'),'normalizer_sha256':sha256(exp/'normalize.py'),
  'reviewed_first_other_feedback_differences':len(records),'categories':dict(Counter(r['category'] for r in records)),
  'pairs':records,'raw_feedback_public':False,'final_test_opened':False,'model_generations_performed':0,'commands_executed':0,
  'scope':'Post-hoc comparison after the frozen v2 projection. After an initial line-order check did not explain the seven residual pairs, the observed truncation-prefix relation was checked on all seven. No claim about unavailable output beyond the original cap. Test-row order remains unchanged in every actual model run; equal line multisets are not a license to reorder arbitrary source or logs.'}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert not a.out.exists();write_json(a.out,review(a.root,a.run))
