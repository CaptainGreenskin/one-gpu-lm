"""CPU-only initial-context screen for every registered failed-state branch."""
import argparse
import json
from pathlib import Path
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.receipts import write_json, sha256
from .failure_continuation import initial_messages, load_snapshot
from .tasks import relative, digest


def screen_identity(p):
    return digest(json.dumps({k:p[k] for k in ['selected_states','evaluation_budget','agent_policy','evaluation_profile_path','evaluation_profile_sha256']},sort_keys=True).encode())


def run(root,p):
    from transformers import AutoTokenizer
    root=Path(root);runtime=json.loads((root/'private/parent-runtime.json').read_text())
    profile=json.loads((root/relative(p['evaluation_profile_path'])).read_text())
    tokenizer=AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False);rows=[]
    for state in p['selected_states']:
        _,prefix,_=load_snapshot(root,state)
        for arm in ['control','correction']:
            messages=initial_messages(prefix,p,arm);selected={}
            def encode(items):
                selected['messages']=items
                return encode_prefix(tokenizer,items,profile['chat_template_kwargs'],p['agent_policy']['format'])[0]
            tokens,context=prepare_context(messages,encode,p['evaluation_budget']['max_input_tokens'],p['agent_policy']['context'])
            contents=[m['content'] for m in selected['messages']]
            intact=prefix['messages'][2*prefix['failed_edit_call']]['content'] in contents and messages[-1]['content'] in contents
            rows.append({'state_id':state['state_id'],'arm':arm,'input_tokens':len(tokens),'failure_context_intact':intact,
                         'input_tokens_sha256':digest(json.dumps(tokens,separators=(',',':')).encode()),'context':context})
    return {'configuration_sha256':screen_identity(p),'passed':bool(rows) and all(r['failure_context_intact'] and r['input_tokens']<=p['evaluation_budget']['max_input_tokens'] for r in rows),
            'rows':rows,'model_generations':0,'optimizer_updates':0,'final_test_opened':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['root','protocol','out']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Preserve previous context screen')
    value=run(args.root,json.loads(args.protocol.read_text()));write_json(args.out,value)
    print(json.dumps({'passed':value['passed'],'branches':len(value['rows']),'model_generations':0}))
