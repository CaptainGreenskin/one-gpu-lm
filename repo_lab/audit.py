"""Summarize private action/observation failures without publishing model text or commands."""
import argparse
from collections import Counter
import json
from pathlib import Path

from training_lab.records import sha256, write_json


def summarize(campaign):
    receipt = json.loads((campaign/'campaign.json').read_text())
    if not receipt['complete']:
        raise ValueError('Audit completed campaigns only')
    rows=[]
    for result in receipt['results']:
        messages=json.loads((campaign/'private'/result['id']/'trajectory.json').read_text())['messages']
        formats=Counter();observations=Counter()
        for message in messages:
            extra=message.get('extra',{})
            if message['role']=='user' and 'tool' in extra:
                observations[str(extra['tool']['returncode'])]+=1
            if message['role']!='assistant' or extra.get('actions'):
                continue
            text=message['content']
            if extra.get('generation',{}).get('hit_generation_limit'):
                kind='truncated_at_generation_limit'
            elif '<function=Bash>' in text:
                kind='unsupported_tool_name_case'
            elif '<tool_call>' in text and '<parameter=command>' not in text:
                kind='unsupported_or_malformed_parameter'
            elif '<tool_call>' in text:
                kind='other_unsupported_native_action'
            else:
                kind='no_supported_action'
            formats[kind]+=1
        rows.append({'id':result['id'],'split':result['split'],'passed':result['passed'],
            'exit_status':result['exit_status'],'grading_reason':result['reason'],
            'format_error_categories':dict(formats),'tool_returncode_counts':dict(observations),
            'generation_limit_hits':sum(g['hit_generation_limit'] for g in result['generations'])})
    return {'schema_version':1,'campaign_sha256':sha256(campaign/'campaign.json'),'tasks':rows,
        'format_categories':dict(sum((Counter(r['format_error_categories']) for r in rows),Counter())),
        'interpretation':'Heuristic classification of observed interface failures, not a causal diagnosis or intrinsic coding score. Nonzero tool status can be an expected grep/test result. Raw messages and commands are excluded.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    write_json(args.out,summarize(args.campaign))
