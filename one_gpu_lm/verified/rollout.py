"""Collect actual local-model actions on admitted training tasks, with clean grading."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import system_prompt,format_feedback
from one_gpu_lm.common.receipts import write_json
from one_gpu_lm.serving.client import request
from .tasks import prepare,grade,files,digest
from .sandbox import Sandbox


def atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');write_json(temp,value)
    with temp.open('rb') as stream:os.fsync(stream.fileno())
    temp.replace(path)


def normalization(repo):
    exp=Path(repo)/'experiments/024-feedback-noise';sys.path.insert(0,str(exp))
    spec=importlib.util.spec_from_file_location('verified_feedback024',exp/'normalize.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run(root,data,task,source,backend,sandbox,protocol,out):
    root,data,out=Path(root),Path(data),Path(out)
    if out.exists():raise ValueError('Preserve previous trajectory; no implicit retry')
    if task['split']!='training':raise ValueError('Teacher may only see training tasks')
    out.mkdir(parents=True)
    budget=protocol['teacher_budget'];policy=protocol['agent_policy']
    norm=normalization(root);snapshot=data/'snapshots'/task['repository'].replace('/','--')
    before=prepare(snapshot,out/'workspace',source,task)
    prompt=system_prompt(policy).replace('at most 12 model calls',f"at most {budget['max_model_calls']} model calls")
    prompt+='\nRun the declared regression command as a separate command before submitting. Hidden checks will evaluate the submitted source independently.'
    instance=(f"Task: {task['description']}\nAllowed source files: {', '.join(task['editable_files'])}\n"
              f"Regression command: {task['regression_command']}\nOnly allowed source changes are submitted to a fresh grader. Preserve public APIs and passing behavior.")
    messages=[{'role':'system','content':prompt},{'role':'user','content':instance}]
    started=time.monotonic();epoch=time.time();health=request(backend,{'operation':'health'})
    if health['model_id']!=protocol['model_id'] or health['revision']!=protocol['revision'] or health['adapter'] or health['profile_sha256']!=protocol['teacher_profile_sha256']:
        raise ValueError('Unexpected teacher artifact')
    state={'schema_version':1,'id':task['id'],'repository':task['repository'],'family':task['family'],'split':'training',
           'producer_kind':'model','teacher_kind':'local_higher_budget_self_distillation_pilot',
           'backend':health,'protocol_sha256':digest(json.dumps(protocol,sort_keys=True).encode()),
           'started_epoch':epoch,'complete':False,'boundary':'before_generation','messages':messages,'steps':[],
           'generated_tokens':0,'attempted_calls':0,'submitted':False,'terminal_reason':None,'runtime_variant':'current',
           'original_source_sha256':{n:before[n] for n in task['editable_files']}}
    atomic(out/'trajectory.json',state)
    last_check_state=None
    for call in range(1,budget['max_model_calls']+1):
        remaining_time=budget['max_task_seconds']-(time.monotonic()-started)
        remaining_tokens=budget['max_total_generated_tokens']-state['generated_tokens']
        if remaining_time<=0 or remaining_tokens<=0:
            state['terminal_reason']='time_limit' if remaining_time<=0 else 'token_limit';break
        payload={'operation':'generate','messages':messages,'agent_policy':policy,
                 'max_input_tokens':budget['max_input_tokens'],'max_new_tokens':min(remaining_tokens,budget['max_generated_tokens_per_call'])}
        state.update(boundary='in_flight',attempted_calls=call);atomic(out/'trajectory.json',state)
        request_path=out/f'request-{call:03d}.json';write_json(request_path,payload)
        try:
            generated=request(backend,payload,timeout=max(1,min(180,remaining_time)))
        except Exception as error:
            state.update(terminal_reason='backend_error',backend_error_type=type(error).__name__,unknown_inflight_generation_cost=True)
            atomic(out/'trajectory.json',state)
            raise RuntimeError('Stop pilot after an unaccounted backend request') from None
        state['generated_tokens']+=generated['generated_tokens']
        step={'call':call,'request_sha256':digest(request_path.read_bytes()),'generation':generated,
              'source_before':{n:digest((out/'workspace'/n).read_bytes()) if (out/'workspace'/n).is_file() and not (out/'workspace'/n).is_symlink() else None for n in task['editable_files']},'action_format_valid':False}
        assistant={'role':'assistant','content':generated['text'],'extra':{}}
        messages.append(assistant)
        try:
            action=parse_action(generated['text']);step['action_format_valid']=True;step['command']=action['command']
            assistant['extra']['actions']=[action]
        except ValueError:
            messages.append({'role':'user','content':format_feedback(policy)})
            state['steps'].append(step);state['boundary']='completed_step';atomic(out/'trajectory.json',state);continue
        remaining_time=budget['max_task_seconds']-(time.monotonic()-started)
        if remaining_time<=0:
            state['steps'].append(step);state['terminal_reason']='time_limit_before_tool';break
        sandbox.timeout=max(1,min(budget['max_tool_seconds'],int(remaining_time)))
        output=sandbox.execute(out/'workspace',action['command'])
        write_json(out/f'tool-{call:03d}.json',output)
        raw=output['output'];cap=budget['max_tool_output_chars']
        prefix=raw if len(raw)<=cap else raw[:cap]+norm.MARKER
        delivered,receipt=norm.transform(prefix,'extended',cap)
        step['observation']={k:v for k,v in output.items() if k!='output'}
        step['normalization']=receipt
        try:
            after=files(out/'workspace');step['source_after']={n:after.get(n) for n in task['editable_files']}
        except ValueError:
            state['steps'].append(step);state['terminal_reason']='unsafe_workspace';break
        observation={'role':'user','content':f"Exit code: {output['returncode']}\n{delivered}",
                     'extra':{'tool':step['observation']}}
        messages.append(observation)
        if action['command']==task['regression_command'] and output['returncode']==0:last_check_state=step['source_after']
        state['steps'].append(step);state['boundary']='completed_step';atomic(out/'trajectory.json',state)
        if action['command']=='echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' and output['returncode']==0:
            state['submitted']=True;state['terminal_reason']='submitted';break
    if state['terminal_reason'] is None:state['terminal_reason']='call_limit'
    def independent_execute(workspace,command):
        sandbox.timeout=budget['max_tool_seconds'];value=sandbox.execute(workspace,command)
        write_json(out/'grader-output.json',value);return value
    try:
        result=grade(snapshot,out/'grader',source,task,out/'workspace',before,independent_execute)
    except ValueError:
        result={'passed':False,'reason':'unsafe_workspace'}
    final_state={n:digest((out/'workspace'/n).read_bytes()) for n in task['editable_files'] if (out/'workspace'/n).is_file() and not (out/'workspace'/n).is_symlink()}
    state.update(complete=True,boundary='finished',grade=result,finished_epoch=time.time(),seconds=time.monotonic()-started,
                 agent_verified_final_source=last_check_state is not None and last_check_state==final_state,
                 final_source_sha256=final_state)
    state['eligible_verified_trajectory']=result['passed'] and state['submitted'] and state['agent_verified_final_source']
    atomic(out/'trajectory.json',state)
    # Public candidate receipts deliberately omit tasks, commands, messages and paths.
    public={k:state[k] for k in ['id','repository','family','split','producer_kind','teacher_kind','complete','generated_tokens','attempted_calls','submitted','terminal_reason','seconds','agent_verified_final_source','eligible_verified_trajectory','original_source_sha256','final_source_sha256']}
    public.update(grade=result,trajectory_sha256=digest((out/'trajectory.json').read_bytes()),
                  invalid_actions=sum(not s['action_format_valid'] for s in state['steps']),
                  first_edit_call=next((s['call'] for s in state['steps'] if s.get('source_after') and s['source_after']!=s['source_before']),None),
                  failed_tool_commands=sum(s.get('observation',{}).get('returncode',0)!=0 for s in state['steps']))
    write_json(out/'result.json',public)
    return public
