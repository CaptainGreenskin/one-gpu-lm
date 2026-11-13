"""Read-only post-hoc classification of residual feedback; never model input.

This diagnostic was added after the first raw/canonical task pair showed a
Python test-duration difference with identical actions and grades. It does not
change the preregistered normalizer, analysis, budgets, or executed tasks.
Only hashes, counts and candidate difference classes leave private journals.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

EXP=Path(__file__).resolve().parent
sys.path.insert(0,str(EXP.parents[1]));sys.path.insert(0,str(EXP))
from one_gpu_lm.common.receipts import sha256,write_json
from analysis import summarize
from normalize import digest

UNITTEST=re.compile(r'^(Ran \d+ tests? in )\d+(?:\.\d+)?s(\r?)$',re.MULTILINE)
PYTEST=re.compile(r'^(=+ [^\r\n]+ in )\d+(?:\.\d+)?s( =+\r?)$',re.MULTILINE)
TEST_COUNT=r'\d+ (?:passed|failed|skipped|xfailed|xpassed|deselected|errors?|warnings?)'
PYTEST_QUIET=re.compile(r'^('+TEST_COUNT+r'(?:, '+TEST_COUNT+r')* in )\d+(?:\.\d+)?s(\r?)$',re.MULTILINE)
OBJECT_ADDRESS=re.compile(r'( at )0x[0-9a-fA-F]+(?=>)')

def mask(text,kind):
    if kind=='test_duration':
        text,n=UNITTEST.subn(r'\g<1><duration>s\2',text)
        text,m=PYTEST.subn(r'\g<1><duration>s\2',text)
        text,q=PYTEST_QUIET.subn(r'\g<1><duration>s\2',text)
        return text,n+m+q
    return OBJECT_ADDRESS.subn(r'\1<address>',text)

def classify(a,b):
    if a==b:raise ValueError('Residual review requires different delivered text')
    kinds=['test_duration','object_repr_address']
    counts={};normalized={}
    for kind in kinds:
        x,n=mask(a,kind);y,m=mask(b,kind)
        counts[kind]={'a':n,'b':m};normalized[kind]=(x,y)
        if x==y:return {'candidate_class':kind+'_only','masked_field_counts':counts}
    x=mask(normalized['test_duration'][0],'object_repr_address')[0]
    y=mask(normalized['test_duration'][1],'object_repr_address')[0]
    return {'candidate_class':'test_duration_and_object_repr_address' if x==y else 'unclassified_text_change',
            'masked_field_counts':counts}

def review(root,run):
    root=Path(root);run=Path(run);read=lambda p:json.loads(p.read_text())
    audit=read(run/'execution-audit.json')
    campaigns={p.parent.name:read(p) for p in (run/'campaigns').glob('*/campaign.json')}
    value=summarize(campaigns,audit);rows=[]
    for pair in value['pairs']:
        if pair['category']!='other_tool_text':continue
        texts=[];hashes=[]
        for key in ['a','b']:
            task=next(t for t in audit['campaigns'][pair[key]]['tasks'] if t['id']==pair['id'])
            observation=next(t for t in task['observation_receipts'] if t['model_call']==pair['call'])
            path=run/'campaigns'/pair[key]/'private'/pair['id']/'observations.jsonl'
            if sha256(path)!=task['journal_sha256']:raise ValueError('Private journal changed since execution audit')
            journal=[json.loads(s) for s in path.read_text().splitlines()]
            entry=journal[observation['call_index']-1]
            if digest(entry['delivered_output'])!=observation['delivered_sha256']:raise ValueError('Delivered feedback mismatch')
            texts.append(entry['delivered_output']);hashes.append(observation['delivered_sha256'])
        rows.append({k:pair[k] for k in ['comparison','model','condition','a','b','id','call','outcome_disagreement']}|
                    {'delivered_sha256':hashes,'delivered_characters':list(map(len,texts)),**classify(*texts)})
    return {'schema_version':1,'post_hoc':True,'preregistered_metric':False,
            'source_sha256':sha256(Path(__file__)),'execution_audit_sha256':sha256(run/'execution-audit.json'),
            'analysis_sha256':sha256(EXP/'analysis.py'),'reviewed_first_other_feedback_differences':len(rows),
            'candidate_classes':dict(Counter(r['candidate_class'] for r in rows)),'pairs':rows,
            'model_generations_performed':0,'commands_executed':0,'normalizer_changed':False,
            'raw_feedback_public':False,'final_test_opened':False,
            'interpretation':'Post-hoc text-equivalence labels identify candidate fields at the first observed residual-feedback boundary. They do not establish that a field caused action or grade changes, and do not cover all differences later in a trace.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError('Preserve the existing diagnostic receipt')
    write_json(args.out,review(args.root,args.run))
