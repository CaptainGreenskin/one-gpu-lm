"""Before024 generations, audit additive transforms on every private023 output."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json
from _feedback024 import module
norm=module('normalize')

def review(parent):
    parent=Path(parent);run=parent/'runs/stability023'
    audit=json.loads((run/'execution-audit.json').read_text())
    assert json.loads((run/'postrun-status.json').read_text())['complete']
    checked=changed=0;fields=Counter();records=[];transformed={}
    for name,cell in audit['campaigns'].items():
        for task in cell['tasks']:
            p=run/'campaigns'/name/'private'/task['id']/'observations.jsonl'
            assert sha256(p)==task['journal_sha256']
            journal=[json.loads(line) for line in p.read_text().splitlines()]
            assert len(journal)==len(task['observation_receipts'])
            for entry,observed in zip(journal,task['observation_receipts']):
                raw=entry['raw_output'];baseline,a=norm.transform(raw,'baseline');extended,b=norm.transform(raw,'extended')
                assert baseline==norm.legacy.transform(raw,'canonical')[0]
                assert norm.digest(raw)==observed['raw_sha256'] and a['baseline_sha256']==observed['canonical_sha256']
                assert len(extended)<=len(baseline) and a['baseline_sha256']==b['baseline_sha256']
                checked+=1;changed+=b['additional_changed'];fields.update({k:v for k,v in b['eligible_fields'].items() if k not in ['ls_timestamp','js_summary_duration']})
                transformed[(name,task['id'],observed['model_call'])]=b
            records.append({'cell':name,'id':task['id'],'observations':len(journal),'journal_sha256':sha256(p)})
    residual=json.loads((run/'feedback-residual-review.json').read_text())
    resolved=[]
    for pair in residual['pairs']:
        a,b=[transformed[(pair[k],pair['id'],pair['call'])] for k in ['a','b']]
        resolved.append({'a':pair['a'],'b':pair['b'],'id':pair['id'],'call':pair['call'],
                         'old_residual_class':pair['candidate_class'],'v2_feedback_equal':a['canonical_sha256']==b['canonical_sha256']})
    assert checked==sum(len(t['observation_receipts']) for c in audit['campaigns'].values() for t in c['tasks'])==2626
    return {'schema_version':1,'passed':True,'source_sha256':sha256(Path(__file__)),
      'normalizer_sha256':sha256(Path(__file__).with_name('normalize.py')),'baseline_normalizer_sha256':sha256(Path(norm.legacy.__file__)),
      'parent_execution_audit_sha256':sha256(run/'execution-audit.json'),'observations_checked':checked,
      'baseline_exactly_matches_023':True,'additional_changed_observations':changed,'additional_eligible_fields':dict(fields),
      'old_first_residual_pairs':len(resolved),'old_first_residual_pairs_now_equal':sum(p['v2_feedback_equal'] for p in resolved),
      'pairs':resolved,'journals':records,'model_generations':0,'commands_executed':0,'final_test_opened':False,
      'scope':'CPU transformation checks on existing023 observations,not024 model scores. Equalizing an old feedback prefix does not predict later024 actions or grades.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();write_json(a.out,review(a.parent))
