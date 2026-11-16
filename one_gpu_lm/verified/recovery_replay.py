"""Replay mixed student-prefix/base-continuation inputs without rerunning commands."""
import argparse
import json
from pathlib import Path
import traceback

from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import system_prompt, format_feedback, prepare_context
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.receipts import sha256, write_json
from .rollout import normalization
from .submission import SUBMIT
from .tasks import digest, files, prepare, grade, relative, validate_source


def initial_messages(task, protocol):
    budget=protocol['evaluation_budget'];policy=protocol['agent_policy']
    prompt=system_prompt(policy).replace('at most 12 model calls',f"at most {budget['max_model_calls']} model calls")
    prompt+='\nRun the declared regression command as a separate command before submitting. Hidden checks will evaluate the submitted source independently.'
    instance=(f"Task: {task['description']}\nAllowed source files: {', '.join(task['editable_files'])}\n"
              f"Regression command: {task['regression_command']}\nOnly allowed source changes are submitted to a fresh grader. Preserve public APIs and passing behavior.")
    return [{'role':'system','content':prompt},{'role':'user','content':instance}]


def replay(directory, trace, task, protocol, profile, tokenizer, normalizer):
    directory=Path(directory);budget=protocol['evaluation_budget'];policy=protocol['agent_policy']
    messages=initial_messages(task,protocol);total=0;input_total=0;checks=[];actions=[];feedback=[]
    errors=[];last_source=trace['original_source_sha256'];verified=None
    guard=True
    expected_roles = {'student'} | ({'base'} if trace['branch']=='teacher' else set())
    from .recovery_rollout import identity_ok
    identity_ok_all = (set(trace['backend_identities'])==expected_roles and all(
        identity_ok(value, protocol, role) for role,value in trace['backend_identities'].items()))
    for index,step in enumerate(trace['steps'],1):
        flags={};call=step['call']
        student = trace['branch']!='teacher' or call<=3
        flags['producer_role'] = step['model_role']==('student' if student else 'base')
        try:
            request_path=directory/f'request-{call:03d}.json';payload=json.loads(request_path.read_text())
            expected={'operation':'generate','messages':messages,'agent_policy':policy,
                      'max_input_tokens':budget['max_input_tokens'],
                      'max_new_tokens':min(budget['max_total_generated_tokens']-total,budget['max_generated_tokens_per_call'])}
            flags['registered_request']=call==index and payload==expected and sha256(request_path)==step['request_sha256']
            generated=step['generation']
            def encode(items):
                return encode_prefix(tokenizer,items,profile['chat_template_kwargs'],policy['format'])[0]
            tokens,context=prepare_context(payload['messages'],encode,payload['max_input_tokens'],policy['context'])
            flags['token_input_replay']=(len(tokens)==generated['input_tokens']
                 and digest(json.dumps(tokens,separators=(',',':')).encode())==generated['input_tokens_sha256']
                 and context==generated['context'])
            flags['generated_receipt']=(len(generated['ids'])==generated['generated_tokens']
                 and tokenizer.decode(generated['ids'],skip_special_tokens=True)==generated['text']
                 and generated.get('adapter') is student
                 and generated['generated_tokens']<=payload['max_new_tokens']
                 and generated['input_tokens']<=budget['max_input_tokens'])
            total+=generated['generated_tokens'];input_total+=generated['input_tokens']
            flags['source_chain']=step['source_before']==last_source
            assistant={'role':'assistant','content':generated['text'],'extra':{}}
            messages.append(assistant)
            try:
                action=parse_action(generated['text'])
            except ValueError:
                action=None
            flags['action_parse']=(action is not None)==step['action_format_valid']
            if action is None:
                messages.append({'role':'user','content':format_feedback(policy)})
                actions.append({'valid':False,'generation_sha256':digest(generated['text'].encode())})
                flags['no_execution_for_malformed_action']='observation' not in step
            else:
                command=action['command'];assistant['extra']['actions']=[action]
                flags['recorded_command']=command==step['command']
                actions.append({'valid':True,'command_sha256':digest(command.encode())})
                if 'observation' not in step:
                    flags['unexecuted_due_to_actor_stop']=(index==len(trace['steps']) and trace['terminal_reason']=='time_limit_before_tool')
                else:
                    raw=json.loads((directory/f'tool-{call:03d}.json').read_text())
                    flags['raw_tool_receipt']={k:v for k,v in raw.items() if k!='output'}==step['observation']
                    prefix=raw['output'];cap=budget['max_tool_output_chars']
                    if len(prefix)>cap:prefix=prefix[:cap]+normalizer.MARKER
                    delivered,receipt=normalizer.transform(prefix,'extended',cap)
                    flags['feedback_normalization']=receipt==step['normalization']
                    if guard:
                        before=step['source_before']
                        if verified!=before:verified=None
                        reject=(command==SUBMIT and (not all(v is not None for v in before.values()) or verified!=before))
                        flags['guard_execution']=raw.get('executed') is (not reject)
                        if reject:
                            text=('Submission rejected by the collection policy. Run exactly the following regression command '
                                  'as the entire next bash tool call, without a cd prefix or additional commands: '
                                  +task['regression_command']+'\nAfter it passes on the unchanged source, submit again.')
                            flags['guard_rejection']=(raw['returncode']==-2 and raw['reason']=='submission_requires_verification'
                                  and raw['policy_event']=='rejected_submit_without_fresh_exact_regression' and raw['output']==text)
                    else:
                        flags['plain_runtime_not_intercepted']='executed' not in raw and 'policy_event' not in raw
                    if 'source_after' in step:
                        after=step['source_after']
                        if guard:
                            if after!=step['source_before'] or (command==task['regression_command'] and raw['returncode']!=0):verified=None
                            if command==task['regression_command'] and raw['returncode']==0 and after==step['source_before'] and all(v is not None for v in after.values()):verified=after
                        last_source=after
                        messages.append({'role':'user','content':f"Exit code: {raw['returncode']}\n{delivered}",'extra':{'tool':step['observation']}})
                        feedback.append(digest(messages[-1]['content'].encode()))
                    else:
                        flags['unsafe_workspace_terminal']=index==len(trace['steps']) and trace['terminal_reason']=='unsafe_workspace'
            checks.append({'call':call,'checks':flags,'passed':all(flags.values())})
        except Exception as error:
            errors.append({'call':call,'error_type':type(error).__name__})
            checks.append({'call':call,'checks':flags,'passed':False})
    request_names={p.name for p in directory.glob('request-*.json')}
    expected_names={f'request-{n:03d}.json' for n in range(1,trace['attempted_calls']+1)}
    complete_cost=(len(trace['steps'])==trace['attempted_calls'] and request_names==expected_names
                   and total==trace['generated_tokens'] and not trace.get('unknown_inflight_generation_cost',False))
    passed=(identity_ok_all and not errors and all(c['passed'] for c in checks) and complete_cost
            and messages==trace['messages'] and trace['attempted_calls']<=budget['max_model_calls']
            and total<=budget['max_total_generated_tokens'])
    return {'passed':passed,'identity_passed':identity_ok_all,'complete_generation_accounting':complete_cost,
            'reconstructed_verified_source':verified,'reconstructed_final_source':last_source,
            'complete_message_history_reconstructed':messages==trace['messages'],'checks':checks,'errors':errors,
            'known_generated_tokens':sum(s['generation'].get('generated_tokens',0) for s in trace['steps']),
            'known_input_tokens':sum(s['generation'].get('input_tokens',0) for s in trace['steps']),
            'action_fingerprint':digest(json.dumps(actions,sort_keys=True).encode()),
            'action_step_fingerprints':[digest(json.dumps(a,sort_keys=True).encode()) for a in actions],
            'delivered_feedback_fingerprints':feedback}
