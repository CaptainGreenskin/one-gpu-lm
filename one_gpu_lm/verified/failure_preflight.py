"""CPU-only initial-context screen for every registered failed-state branch."""
import argparse
import json
from pathlib import Path
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.reasoning import request_options, settings, separate
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.common.receipts import write_json, sha256
from .failure_continuation import initial_messages, load_snapshot
from .tasks import relative, digest


def screen_identity(p):
    values={k:p[k] for k in ['selected_states','evaluation_budget','agent_policy','evaluation_profile_path','evaluation_profile_sha256']}
    if 'continuation_policy' in p:values['continuation_policy']=p['continuation_policy']
    if 'reasoning_comparison' in p:values['reasoning_comparison']=p['reasoning_comparison']
    return digest(json.dumps(values,sort_keys=True).encode())


def run(root,p):
    from transformers import AutoTokenizer
    root=Path(root);runtime=json.loads((root/'private/parent-runtime.json').read_text())
    profile=json.loads((root/relative(p['evaluation_profile_path'])).read_text())
    tokenizer=AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False);rows=[]
    for state in p['selected_states']:
        _,prefix,_=load_snapshot(root,state)
        for arm in ['control','correction']:
            messages=initial_messages(prefix,p,arm);selected={}
            options=request_options(p,arm)
            template_kwargs=settings(profile,options)[0]
            def encode(items):
                selected['messages']=items
                return encode_prefix(tokenizer,items,template_kwargs,p['agent_policy']['format'])[0]
            tokens,context=prepare_context(messages,encode,p['evaluation_budget']['max_input_tokens'],p['agent_policy']['context'])
            contents=[m['content'] for m in selected['messages']]
            intact=prefix['messages'][2*prefix['failed_edit_call']]['content'] in contents and messages[-1]['content'] in contents
            rows.append({'state_id':state['state_id'],'arm':arm,'input_tokens':len(tokens),'failure_context_intact':intact,
                         'input_tokens_sha256':digest(json.dumps(tokens,separators=(',',':')).encode()),'context':context})
    proof=[]
    if p.get('reasoning_comparison'):
        student=json.loads((root/relative(p['reasoning_comparison']['student_profile_path'])).read_text())
        for arm in ['control','correction']:
            options=request_options(p,arm);kwargs=settings(profile,options)[0]
            sample=[{'role':'user','content':'Inspect the current directory.'}]
            tokens,_=encode_prefix(tokenizer,sample,kwargs,p['agent_policy']['format'])
            rendered=tokenizer.decode(tokens,skip_special_tokens=False)
            ending='<think>\n' if kwargs['enable_thinking'] else '<think>\n\n</think>\n\n'
            if not rendered.endswith(ending):raise ValueError('Unexpected template thought prefill')
            target='<tool_call>\n<function=bash>\n<parameter=command>pwd</parameter>\n</function>\n</tool_call>'
            text=('Inspect. </think>\n' if kwargs['enable_thinking'] else '')+target
            ids=tokenizer.encode(text,add_special_tokens=False);split=separate(tokenizer,ids,profile,options)
            if not split['valid'] or split['action_text'].strip()!=target:raise ValueError('Action separation screen failed')
            encoded=encode_training_example({'messages':sample,'target':split['action_text']},tokenizer,student)
            proof.append({'arm':arm,'prefill_verified':True,'separation_verified':True,'student_target_encoded':True,'supervised_tokens':encoded['supervised_tokens'],'template_kwargs':kwargs})
    return {'variant_preflight':proof,'configuration_sha256':screen_identity(p),'passed':bool(rows) and all(r['failure_context_intact'] and r['input_tokens']<=p['evaluation_budget']['max_input_tokens'] for r in rows),
            'rows':rows,'model_generations':0,'optimizer_updates':0,'final_test_opened':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['root','protocol','out']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Preserve previous context screen')
    value=run(args.root,json.loads(args.protocol.read_text()));write_json(args.out,value)
    print(json.dumps({'passed':value['passed'],'branches':len(value['rows']),'model_generations':0}))
