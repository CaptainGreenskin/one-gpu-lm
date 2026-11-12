"""Audit completed collection prefixes and exact derived inputs; never execute tools."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import validate_policy


def audit(root, run, collection_only=False, allow_partial=False):
    root, run = Path(root).resolve(), Path(run).resolve()
    read = lambda p: json.loads(Path(p).read_text())
    exp = root/'experiments/021-observed-recovery'; protocol=read(exp/'protocol.json')
    profile=read(exp/'profile-observed.json'); runtime=read(root/'runtime.json')
    tasks={t['id']:t for t in read(root/'private/tasks.json')['tasks']}
    train={k:t for k,t in tasks.items() if t['split']=='train'}
    admitted={t['id']:t for t in read(root/'runs/admission/admission.json')['tasks']}
    collection=run/'collection'; collected=read(collection/'collection.json')
    assert len(train)==55 and len({r['id'] for r in collected['tasks']})==len(collected['tasks'])
    assert {r['id'] for r in collected['tasks']}<=set(train)
    if not allow_partial: assert collected['complete'] and len(collected['tasks'])==55
    else: assert collection_only and collected['tasks']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False)
    state_index={}; records=[]
    for item in collected['tasks']:
        task=train[item['id']]; private=collection/item['id']; n=0; seen={}
        for name,digest in sorted(item['states'].items()):
            path=private/'states'/name; assert sha256(path)==digest
            state=read(path); used=state['completed_calls']; assert used==n+1 and 1<=used<=8;n=used
            assert state['task_id']==task['id'] and state['split']=='train'
            identity=state['identity'];health=identity['backend']
            assert identity['task_spec_sha256']==protocol['task_spec_sha256']
            assert identity['protocol_sha256']==sha256(exp/'protocol.json')
            assert identity['agent_policy']==validate_policy(protocol['variants']['shared'])
            assert health['model_id']==protocol['model_id'] and health['revision']==protocol['revision']
            assert health['profile_sha256']==sha256(exp/'profile-observed.json') and not health['adapter'] and health['adapter_sha256']=={}
            if state['sources']:
                hashes=sorted((name,hashlib.sha256(text.encode()).hexdigest()) for name,text in state['sources'].items())
                assert hashlib.sha256(json.dumps(hashes,separators=(',',':')).encode()).hexdigest()==state['source_state']['editable_state_sha256']
            seen[used]=state;state_index[digest]=state
        checkpoint=private/'checkpoint.json'
        if checkpoint.exists():
            boundary=read(checkpoint)
            if boundary.get('boundary')=='completed_step':
                assert boundary['n_calls']==n,'A completed checkpoint lacks captured state evidence'
            else:
                assert item['timed_out'] or item['returncode'] not in {0,75},'Successful collection left an in-flight request'
        elif item['returncode'] in {0,75}:
            raise ValueError('Successful collection lacks its checkpoint')
        verified=0
        if seen:
            trajectory=seen[n]['trajectory'];local=trajectory['local_model'];messages=[m for m in trajectory['messages'] if m['role']!='exit']
            original=read(private/'original.json')
            assert all(original[p]==admitted[task['id']]['broken_sha256'][p] for p in task['editable_files'])
            source_hashes=sorted((p,original[p]) for p in task['editable_files'])
            current={'changed_editable_files':0,'invalid_editable_files':0,
                     'editable_state_sha256':hashlib.sha256(json.dumps(source_hashes,separators=(',',':')).encode()).hexdigest()}
            call=step=0
            def check(prefix,response):
                tracker=object.__new__(EditableProgress);tracker.snapshot=lambda:dict(current)
                packed,progress=tracker.prepare_request(prefix,call,12,True)
                assert progress==response['progress']
                tokens,dialogue=encode_prefix(tokenizer,packed,profile['chat_template_kwargs'],'native_roundtrip')
                assert len(tokens)==response['input_tokens']
                assert hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()==response['input_tokens_sha256']
                assert dialogue==response['dialogue']
            for index,message in enumerate(messages):
                if message['role']=='assistant':
                    response=local['responses'][call];assert message['extra']['generation']==response
                    check(messages[:index],response)
                    assert response['generated_tokens']<=512 and response['input_tokens']<=8192
                    try: action=parse_action(message['content']);valid=True
                    except ValueError:valid=False
                    assert valid==response['action_format_valid']
                    if valid:assert message['extra']['actions']==[action]
                    call+=1
                elif 'tool' in message.get('extra',{}):
                    observed=local['source_progress'][step]; assert observed['call_index']==call
                    current={k:v for k,v in observed.items() if k!='call_index'}
                    assert current==seen[call]['source_state'];step+=1
            for response in local.get('backend_errors',[]):
                check(messages,response);call+=1
            assert call==n and step==len(local['source_progress']) and local['generated_tokens']<=6144
            verified=call
        records.append({'id':task['id'],'completed_states':n,'verified_completed_requests':verified,
                        'returncode':item['returncode'],'timed_out':item['timed_out']})
    if collection_only:
        return {'passed':True,'complete_collection':collected['complete'],'tasks_checked':len(records),
                'completed_requests_verified':sum(r['verified_completed_requests'] for r in records),
                'audit_source_sha256':sha256(Path(__file__)),'read_only_no_commands_executed':True,
                'collection_status_counts':dict(Counter(str(r['returncode']) for r in records))}
    derived=root/'runs/reference-data021';receipt=read(derived/'observed/receipt.json')
    public=read(derived/'derived-data.json');prepared=root/'runs/prepared-data021/observed'
    assert sha256(prepared/'train.jsonl')==public['prepared_data_sha256']
    parent=[json.loads(s) for s in (root/'runs/prepared-data020/recovery/train.jsonl').read_text().splitlines()]
    rows=[json.loads(s) for s in (prepared/'train.jsonl').read_text().splitlines()]
    assert len(rows)==len(parent)==330 and [r['id'] for r in rows]==[r['id'] for r in parent]
    changed=Counter();common=fallback=0;supervised=inputs=0
    for old,row in zip(parent,rows):
        p=row['provenance'];assert p['split']=='train' and p['task_id'] in train
        enc=encode_training_example(row,tokenizer,profile)
        assert len(enc['ids'])<=4096 and enc['supervised_tokens']<=512
        assert enc['labels'][:enc['prefix_tokens']]==[-100]*enc['prefix_tokens']
        supervised+=enc['supervised_tokens'];inputs+=len(enc['ids'])
        if p['stage'] in ['inspect','repair','verify','submit']:
            assert row==old;common+=1
        elif p.get('curriculum')!='021-observed-recovery':
            assert row==old;fallback+=1
        else:
            state=state_index[p['observed_state_sha256']]
            assert state['task_id']==p['task_id'] and state['completed_calls']==p['observed_calls']
            proof=derived/'private'/p['task_id']/p['stage']/str(p['observed_calls'])/'trajectory.json'
            assert sha256(proof)==p['verified_trajectory_sha256']
            value=read(proof); assert value['grade']['passed'] and value['wrong_grade']['returncode']==1
            messages=value['messages'];index=p['assistant_message_index']
            assert messages[:len(state['trajectory']['messages'])]==state['trajectory']['messages']
            assert messages[index]['content']==row['target'] and messages[index+1]['extra']['tool']['returncode']==0
            progress=object.__new__(EditableProgress);progress.snapshot=lambda:dict(state['source_state'])
            prefix,observed=progress.prepare_request(messages[:index],p['progress']['completed_calls'],12,True)
            assert prefix==row['messages'] and observed==p['progress']
            assert len(value['steps'])<=12 and sum(s.get('supervised_slot') is not None for s in value['steps'])==1
            changed[p['state_category']]+=1
    assert common==220 and sum(changed.values())==receipt['replaced_records']
    assert sum(changed.values())+fallback==110 and dict(changed)==receipt['categories']
    assert supervised==public['supervised_tokens'] and inputs==public['input_tokens']
    for name,digest in protocol['frozen_files'].items():assert sha256(root/name)==digest
    result={'passed':True,'audit_source_sha256':sha256(Path(__file__)),'collection_sha256':sha256(collection/'collection.json'),
            'derived_data_sha256':sha256(derived/'derived-data.json'),'prepared_data_sha256':sha256(prepared/'train.jsonl'),
            'training_tasks':55,'completed_request_prefixes_verified':sum(r['verified_completed_requests'] for r in records),
            'records_checked':330,'common_identical_records':common,'observed_replacements':sum(changed.values()),
            'fallback_records':fallback,'categories':dict(changed),'whole_context_and_masking_verified':True,
            'supervised_tokens':supervised,'input_tokens':inputs,'actual_tool_commands_reexecuted_by_audit':False,
            'scope':'All saved completed training-state prefixes and derived examples; an interrupted in-flight request is not claimed as captured.',
            'collection_status_counts':dict(Counter(str(r['returncode']) for r in records)),
            'raw_traces_public':False}
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--collection-only',action='store_true');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    assert not a.out.exists();value=audit(a.root,a.run,a.collection_only,a.allow_partial);write_json(a.out,value);print(json.dumps(value))
