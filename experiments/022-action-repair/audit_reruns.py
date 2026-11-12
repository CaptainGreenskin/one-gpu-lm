"""Read-only inspection of the first tool feedback in two fixed-020 reruns.

This does not canonicalize model inputs, rerun a model, or establish causality.
Raw feedback remains private; only exact hashes and comparison counts leave.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json

PATTERN=r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+(?:\d{1,2}:\d{2}|\d{4})\b'
IDS=['fast-deep-equal-map-values','fast-deep-equal-regexp-flags']

def inspect(root):
    root=Path(root);read=lambda p:json.loads(p.read_text())
    old=Path(read(root/'private/parent021.json')['root'])
    previous=old/'runs/release021';current=root/'runs/release022'
    audits=[read(p/'execution-audit.json') for p in [previous,current]]
    results=[]
    for ident in IDS:
        records=[];tools=[]
        for folder,audit in zip([previous,current],audits):
            row=next(r for r in audit['variants']['dev-reference020'] if r['id']==ident)
            path=folder/'dev-reference020/private'/ident/'trajectory.json'
            assert sha256(path)==row['trajectory_sha256']
            trajectory=read(path);tool=next(m for m in trajectory['messages'] if 'tool' in m.get('extra',{}))
            records.append(row);tools.append(tool)
        a,b=records;x,y=[v['content'] for v in tools]
        normalize=lambda text:re.sub(PATTERN,'<LS_TIMESTAMP>',text)
        assert a['generation_input_sha256'][0]==b['generation_input_sha256'][0]
        assert a['generation_output_sha256'][0]==b['generation_output_sha256'][0]
        assert a['generation_input_sha256'][1]!=b['generation_input_sha256'][1]
        assert all(t['extra']['tool']['returncode']==0 for t in tools)
        assert x!=y and normalize(x)==normalize(y)
        results.append({'id':ident,'old_trajectory_sha256':a['trajectory_sha256'],
            'current_trajectory_sha256':b['trajectory_sha256'],
            'initial_model_input_equal':True,'initial_model_output_equal':True,'second_input_differs':True,
            'first_tool_returncodes':[t['extra']['tool']['returncode'] for t in tools],
            'first_tool_content_sha256':[hashlib.sha256(t.encode()).hexdigest() for t in [x,y]],
            'first_tool_content_equal':False,'first_tool_content_equal_after_timestamp_normalization':True,
            'timestamp_matches':[len(re.findall(PATTERN,t)) for t in [x,y]],
            'content_characters':[len(t) for t in [x,y]],
            'normalized_content_sha256':hashlib.sha256(normalize(x).encode()).hexdigest()})
    return {'schema_version':1,'passed':True,'audit_source_sha256':sha256(Path(__file__)),
        'prior_execution_audit_sha256':sha256(previous/'execution-audit.json'),
        'current_execution_audit_sha256':sha256(current/'execution-audit.json'),
        'scope':'Only the first tool feedback of these two fixed-020 tasks across 021 and 022.',
        'normalization_pattern':PATTERN,'observations':results,'model_calls_added':0,'inputs_changed':False,
        'raw_feedback_public':False,'causal_effect_tested':False,
        'limitation':'Timestamp text accounts for the observed first feedback difference; no intervention/replay isolates its effect on later actions or final grades.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();assert not a.out.exists();r=inspect(a.root);write_json(a.out,r)
    print(json.dumps({'passed':r['passed'],'tasks':len(r['observations']),'model_calls_added':0}))
