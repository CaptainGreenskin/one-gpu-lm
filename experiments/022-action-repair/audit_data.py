"""Verify all actual training inputs and executed reference receipts without tools."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.evaluation.tasks import apply_mutation
from build_data import assign_modes, locate_command, missing_read_command, repair_and_test_command, MODES, CHANGED_SLOT


def audit(root):
    root=Path(root).resolve();read=lambda p:json.loads(Path(p).read_text())
    exp=root/'experiments/022-action-repair';profile=read(exp/'profile-action.json');plan=read(exp/'preparation-plan.json')
    tasks={t['id']:t for t in read(root/'private/tasks.json')['tasks'] if t['split']=='train'}
    modes=assign_modes(list(tasks.values()))
    admission={t['id']:t for t in read(root/'runs/admission/admission.json')['tasks'] if t['split']=='train'}
    derived=root/'runs/reference-data022';prepared=root/'runs/prepared-data022/action'
    public=read(derived/'derived-data.json');receipt=read(derived/'action/receipt.json')
    assert sha256(prepared/'train.jsonl')==public['prepared_data_sha256']
    assert sha256(prepared/'receipt.json')==public['prepared_receipt_sha256']
    assert sha256(prepared/'selection.json')==public['selection_sha256']
    parent=[json.loads(s) for s in (root/'runs/prepared-data020/recovery/train.jsonl').read_text().splitlines()]
    rows=[json.loads(s) for s in (prepared/'train.jsonl').read_text().splitlines()]
    assert len(rows)==len(parent)==330 and [r['id'] for r in rows]==[r['id'] for r in parent]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(read(root/'runtime.json')['model'],local_files_only=True,trust_remote_code=False)
    common=fallback=inputs=supervised=0;changed=Counter();proofs=[]
    for old,row in zip(parent,rows):
        p=row['provenance'];assert p['split']=='train' and p['task_id'] in tasks
        enc=encode_training_example(row,tokenizer,profile)
        assert len(enc['ids'])<=4096 and 0<enc['supervised_tokens']<=512
        assert enc['labels'][:enc['prefix_tokens']]==[-100]*enc['prefix_tokens']
        inputs+=len(enc['ids']);supervised+=enc['supervised_tokens']
        if p['stage']!=CHANGED_SLOT:
            assert row==old;common+=1;continue
        if p.get('curriculum')!='022-action-repair':
            assert row==old;fallback+=1;continue
        task=tasks[p['task_id']];mode=modes[p['task_id']];assert p['mode']==mode
        repair_call=3 if mode==MODES[0] else 4;assert p['repair_call']==repair_call
        proof=derived/'private'/task['id']/'trajectory.json';assert sha256(proof)==p['verified_trajectory_sha256']
        value=read(proof);steps=value['steps'];messages=value['messages']
        from jinja2 import Template
        from one_gpu_lm.common.actions import INSTANCE
        from one_gpu_lm.common.agent_policy import NATIVE_SYSTEM
        prompt=Template(INSTANCE).render(task=task['description'],editable_files=', '.join(task['editable_files']),
                                         regression_command=task['regression_command'])
        assert messages[:2]==[{'role':'system','content':NATIVE_SYSTEM},{'role':'user','content':prompt}]
        assert len(steps)==repair_call+1<=12
        assert value['grade']['passed'] and not value['wrong_grade']['passed'] and value['wrong_grade']['returncode']==1
        good={name:(root/'sources'/task['repository'].replace('/','--')/name).read_text() for name in task['editable_files']}
        broken=dict(good)
        for change in task['mutations']:broken[change['path']]=apply_mutation(broken[change['path']],change)
        original={name:hashlib.sha256(text.encode()).hexdigest() for name,text in broken.items()}
        assert all(original[n]==admission[task['id']]['broken_sha256'][n] for n in original)
        before={'changed_editable_files':0,'invalid_editable_files':0,
                'editable_state_sha256':hashlib.sha256(json.dumps(sorted(original.items()),separators=(',',':')).encode()).hexdigest()}
        expected=[]
        if mode==MODES[1]:expected.append(('missing_preview_read',missing_read_command(),1))
        expected += [('locate_allowed_sources',locate_command(task),0),
                     ('declared_failing_test',task['regression_command'],1),
                     ('early_repair_and_declared_test',repair_and_test_command(broken,good,task['regression_command']),0),
                     ('submit','echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT',0)]
        assert len(messages)==2+2*len(expected)
        for i,(kind,command,code) in enumerate(expected):
            assistant,tool=messages[2+i*2:4+i*2]
            assert parse_action(assistant['content'])=={'command':command} and assistant['extra']['actions']==[{'command':command}]
            assert steps[i]['kind']==kind and steps[i]['returncode']==tool['extra']['tool']['returncode']==code
            assert steps[i]['call_index']==i+1
            if i<repair_call-1:assert steps[i]['state_before']==steps[i]['state_after']==before
        assert steps[repair_call-1]['supervised_slot']==CHANGED_SLOT
        assert sum(s.get('supervised_slot') is not None for s in steps)==1
        if mode==MODES[1]:assert 'FileNotFoundError' in messages[3]['content']
        index=2+2*(repair_call-1);assert p['assistant_message_index']==index and messages[index]['content']==row['target']
        tracker=object.__new__(EditableProgress);tracker.snapshot=lambda:dict(before)
        prefix,progress=tracker.prepare_request(messages[:index],repair_call-1,12,True)
        assert prefix==row['messages'] and progress==p['progress']
        expected_source={name:hashlib.sha256(text.encode()).hexdigest() for name,text in good.items()}
        assert value['grade']['source_sha256']==expected_source
        assert steps[repair_call-1]['source_sha256_after']==steps[-1]['source_sha256_after']==expected_source
        assert all(sha256(derived/'private'/task['id']/'workspace'/n)==h for n,h in expected_source.items())
        changed[mode]+=1
        proofs.append({'id':task['id'],'mode':mode,'repair_call':repair_call,'trajectory_sha256':sha256(proof),
            'input_and_target_ids_sha256':hashlib.sha256(json.dumps(enc['ids'],separators=(',',':')).encode()).hexdigest(),
            'target_sha256':hashlib.sha256(row['target'].encode()).hexdigest(),'independent_repair_passed':True})
    assert common==275 and sum(changed.values())+fallback==55 and sum(changed.values())==receipt['replaced_records']
    assert dict(changed)==receipt['accepted_modes']
    assert (sum(changed.values())>=plan['data']['minimum_replaced_records'] and
            all(changed[m]>=plan['data']['minimum_each_mode'] for m in MODES)) is receipt['coverage_passed']
    assert inputs==public['input_tokens'] and supervised==public['supervised_tokens']
    return {'schema_version':1,'passed':True,'audit_source_sha256':sha256(Path(__file__)),
        'builder_sha256':sha256(exp/'build_data.py'),'derived_data_sha256':sha256(derived/'derived-data.json'),
        'prepared_data_sha256':sha256(prepared/'train.jsonl'),'records_checked':330,'common_identical_records':common,
        'early_action_replacements':sum(changed.values()),'fallback_records':fallback,'accepted_modes':dict(changed),
        'input_tokens':inputs,'supervised_tokens':supervised,'whole_context_and_masking_verified':True,
        'all_original_failure_recovery_records_preserved':True,'proofs':proofs,
        'actual_commands_reexecuted_by_audit':False,'raw_traces_public':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();assert not a.out.exists();value=audit(a.root);write_json(a.out,value)
    print(json.dumps({k:value[k] for k in ['passed','records_checked','early_action_replacements','fallback_records']}))
