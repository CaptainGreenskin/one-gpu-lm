"""032 repeat rollout, preserving the frozen task and actor rules with durable receipts."""
import json
from pathlib import Path
import time
from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import format_feedback
from one_gpu_lm.common.receipts import write_json
from one_gpu_lm.serving.client import request
from .tasks import files,digest
from .historic_rollout import atomic, normalization
from .historic_tasks import prepare,grade
from .historic_policy import initial_messages, feedback
from one_gpu_lm.serving.receipt_client import generate_once


def run(root,data,task,source,backend,sandbox,protocol,out,arm,*,journal_path,recovery_seconds):
    root,data,out=Path(root),Path(data),Path(out)
    if out.exists():raise ValueError('Preserve previous trajectory; no implicit retry')
    if task['split']!='development':raise ValueError('This evaluator accepts development tasks only')
    if arm not in ['base_current','base_guard','student_current','student_guard']:raise ValueError('Unknown evaluation arm')
    out.mkdir(parents=True)
    budget=protocol['evaluation_budget'];policy=protocol['agent_policy']
    norm=normalization(root);snapshot=data/'snapshots'/task['source_id']
    before=prepare(snapshot,out/'workspace',source,task)
    messages=initial_messages(task,policy,budget)
    started=time.monotonic();epoch=time.time();health=request(backend,{'operation':'health'})
    expected_adapter = protocol['adapter_sha256'] if arm.startswith('student_') else {}
    if (health['model_id']!=protocol['model_id'] or health['revision']!=protocol['revision']
            or health['adapter']!=arm.startswith('student_') or health['profile_sha256']!=protocol['evaluation_profile_sha256']
            or health.get('adapter_sha256',{})!=expected_adapter):
        raise ValueError('Unexpected evaluation artifact')
    state={'schema_version':1,'id':task['id'],'repository':task['repository'],'family':task['family'],'split':'development','arm':arm,'training_export_allowed':False,
           'producer_kind':'model','model_role':'student' if arm.startswith('student_') else 'base',
           'backend':health,'protocol_sha256':digest(json.dumps(protocol,sort_keys=True).encode()),
           'started_epoch':epoch,'complete':False,'boundary':'before_generation','messages':messages,'steps':[],
           'generated_tokens':0,'attempted_calls':0,'submitted':False,'terminal_reason':None,'runtime_variant':'explicit_submission_guard_v1' if arm.endswith('_guard') else 'current',
           'original_source_sha256':{n:before[n] for n in task['editable_files']}}
    atomic(out/'trajectory.json',state)
    if health.get('request_journal_version') != 1:
        raise ValueError('Expected the journaled evaluation backend')
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
            generated, recovery = generate_once(
                backend, journal_path, out.name, call, payload,
                timeout=min(180, remaining_time), recovery_seconds=recovery_seconds)
        except Exception as error:
            state.update(terminal_reason='backend_error',backend_error_type=type(error).__name__,unknown_inflight_generation_cost=True)
            atomic(out/'trajectory.json',state)
            raise RuntimeError('Stop evaluation after an unaccounted backend request') from None
        state['generated_tokens']+=generated['generated_tokens']
        step={'call':call,'request_sha256':digest(request_path.read_bytes()),'generation':generated,
              'source_before':{n:digest((out/'workspace'/n).read_bytes()) if (out/'workspace'/n).is_file() and not (out/'workspace'/n).is_symlink() else None for n in task['editable_files']},'action_format_valid':False}
        if recovery is not None:
            step['transport_timeout_receipt'] = recovery
            state['timeout_call'] = call
            state['accounting_drain_seconds'] = recovery['accounting_drain_seconds']
        assistant={'role':'assistant','content':generated['text'],'extra':{}}
        messages.append(assistant)
        try:
            action=parse_action(generated['text']);step['action_format_valid']=True;step['command']=action['command']
            assistant['extra']['actions']=[action]
        except ValueError:
            messages.append({'role':'user','content':format_feedback(policy)})
            state['steps'].append(step)
            if recovery is not None:
                state['terminal_reason'] = 'time_limit'
                break
            state['boundary']='completed_step';atomic(out/'trajectory.json',state);continue
        remaining_time=budget['max_task_seconds']-(time.monotonic()-started)
        if recovery is not None or remaining_time<=0:
            state['steps'].append(step);state['terminal_reason']='time_limit_before_tool';break
        sandbox.timeout=max(1,min(budget['max_tool_seconds'],int(remaining_time)))
        output=sandbox.execute(out/'workspace',action['command'])
        write_json(out/f'tool-{call:03d}.json',output)
        delivered,receipt=feedback(output['output'],norm,budget['max_tool_output_chars'])
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
    # Independent grading is outside the actor budget. A late patch may still
    # pass that grader, but it must never become an on-budget success.
    actor_seconds=time.monotonic()-started
    state['actor_seconds']=actor_seconds
    state['actor_budget_compliant']=(actor_seconds<=budget['max_task_seconds']
        and state['attempted_calls']<=budget['max_model_calls']
        and state['generated_tokens']<=budget['max_total_generated_tokens']
        and all(s['generation']['generated_tokens']<=budget['max_generated_tokens_per_call']
                and s['generation']['input_tokens']<=budget['max_input_tokens'] for s in state['steps']))
    def independent_execute(workspace,command):
        inputs = out/'grader-input.json'
        atomic(inputs,files(workspace))
        state['grader_input_sha256'] = digest(inputs.read_bytes())
        sandbox.timeout = budget['max_tool_seconds']
        value = sandbox.execute(workspace,command)
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
    state['success_within_budget']=result['passed'] and state['actor_budget_compliant']
    state['complete_success_within_budget']=state['success_within_budget'] and state['submitted']
    atomic(out/'trajectory.json',state)
    # Public candidate receipts deliberately omit tasks, commands, messages and paths.
    public={k:state[k] for k in ['id','repository','family','split','arm','producer_kind','model_role','complete','generated_tokens','attempted_calls','submitted','terminal_reason','seconds','actor_seconds','actor_budget_compliant','success_within_budget','complete_success_within_budget','agent_verified_final_source','eligible_verified_trajectory','original_source_sha256','final_source_sha256']}
    public.update(grade=result,trajectory_sha256=digest((out/'trajectory.json').read_bytes()),
                  invalid_actions=sum(not s['action_format_valid'] for s in state['steps']),
                  first_edit_call=next((s['call'] for s in state['steps'] if s.get('source_after') and s['source_after']!=s['source_before']),None),
                  failed_tool_commands=sum(s.get('observation',{}).get('returncode',0)!=0 for s in state['steps']))
    public.update(timeout_call=state.get('timeout_call'),
                  accounting_drain_seconds=state.get('accounting_drain_seconds',0.0))
    write_json(out/'result.json',public)
    return public
