"""Replay actual development inputs and recorded grading; no generation or rescore."""
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
    guard=trace['arm'].endswith('_guard');student=trace['arm'].startswith('student_')
    expected_adapter=protocol['adapter_sha256'] if student else {}
    identity=trace['backend']
    identity_ok=(identity['model_id']==protocol['model_id'] and identity['revision']==protocol['revision']
                 and identity['profile_sha256']==protocol['evaluation_profile_sha256']
                 and identity['adapter'] is student and identity.get('adapter_sha256',{})==expected_adapter)
    for index,step in enumerate(trace['steps'],1):
        flags={};call=step['call']
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
    passed=(identity_ok and not errors and all(c['passed'] for c in checks) and complete_cost
            and messages==trace['messages'] and trace['attempted_calls']<=budget['max_model_calls']
            and total<=budget['max_total_generated_tokens'])
    return {'passed':passed,'identity_passed':identity_ok,'complete_generation_accounting':complete_cost,
            'complete_message_history_reconstructed':messages==trace['messages'],'checks':checks,'errors':errors,
            'known_generated_tokens':sum(s['generation'].get('generated_tokens',0) for s in trace['steps']),
            'known_input_tokens':sum(s['generation'].get('input_tokens',0) for s in trace['steps']),
            'action_fingerprint':digest(json.dumps(actions,sort_keys=True).encode()),
            'action_step_fingerprints':[digest(json.dumps(a,sort_keys=True).encode()) for a in actions],
            'delivered_feedback_fingerprints':feedback}


def recorded_grade(root, data, directory, task, source, before, destination):
    """Reconstruct recorded grader input, never rerun test commands."""
    snapshot=Path(data)/'snapshots'/task['repository'].replace('/','--')
    def recorded_execute(workspace,command):
        if command!=task['grading_command'] or files(workspace)!=files(Path(directory)/'grader'):
            raise ValueError('Actual independent grader files differ from the submitted patch and frozen checks')
        return json.loads((Path(directory)/'grader-output.json').read_text())
    try:
        return grade(snapshot,destination,source,task,Path(directory)/'workspace',before,recorded_execute)
    except ValueError as error:
        # Unsafe workspaces are a registered ordinary failure; other provenance
        # errors remain audit failures rather than being relabeled as failures.
        try:files(Path(directory)/'workspace')
        except ValueError:return {'passed':False,'reason':'unsafe_workspace'}
        raise error


def run(root, protocol_path, out):
    from transformers import AutoTokenizer
    root,protocol_path,out=Path(root).resolve(),Path(protocol_path).resolve(),Path(out)
    if out.exists():raise ValueError('Preserve the previous evaluation audit')
    protocol=json.loads(protocol_path.read_text());run_dir=root/relative(protocol['run_directory'])
    controller=json.loads((run_dir/'controller.json').read_text())
    supervisor=json.loads((root/relative(protocol['supervisor_receipt'])).read_text())
    if not controller.get('finished_epoch') or not controller.get('backend_stopped') or not supervisor['complete']:
        raise ValueError('Audit only after evaluation processes have stopped')
    runtime=json.loads((root/'private/parent-runtime.json').read_text())
    profile=json.loads((root/relative(protocol['evaluation_profile_path'])).read_text())
    tokenizer=AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False)
    data=root/relative(protocol['data_directory'])
    tasks={t['id']:t for t in json.loads((data/'tasks.json').read_text())['tasks']}
    sources={s['repository']:s for s in json.loads((data/'sources.json').read_text())['sources']}
    norm=normalization(root);before={};audit_private=run_dir/'private/audit-reconstruction'
    if audit_private.exists():raise ValueError('Preserve an interrupted audit reconstruction')
    audit_private.mkdir()
    for task in tasks.values():
        source=sources[task['repository']];snapshot=data/'snapshots'/task['repository'].replace('/','--')
        before[task['id']]=prepare(snapshot,audit_private/'broken'/task['id'],source,task)
    public_rows=[];controller_map={r['run_id']:r for r in controller.get('runs',[])}
    for item in protocol['schedule']:
        task=tasks[item['task_id']];directory=run_dir/'private/tasks'/item['run_id'];path=directory/'trajectory.json'
        row={'id':task['id'],'repository':task['repository'],'family':task['family'],'arm':item['arm'],
             'repetition':item['repetition'],'run_id':item['run_id'],'complete':False,'audit_passed':False}
        if not path.exists():row['reason']='not_run';public_rows.append(row);continue
        trace=json.loads(path.read_text())
        row.update(complete=trace['complete'],attempted_calls=trace['attempted_calls'],generated_tokens=trace['generated_tokens'],
                   seconds=trace.get('seconds',0),unknown_inflight_generation_cost=bool(trace.get('unknown_inflight_generation_cost',False)),
                   trajectory_sha256=sha256(path))
        try:
            replayed=replay(directory,trace,task,protocol,profile,tokenizer,norm)
            row.update(input_tokens=replayed['known_input_tokens'],request_replay=replayed)
            if not trace['complete']:
                row['reason']='partial_trajectory';public_rows.append(row);continue
            result=json.loads((directory/'result.json').read_text())
            expected_grade=recorded_grade(root,data,directory,task,sources[task['repository']],before[task['id']],audit_private/'graders'/item['run_id'])
            budget=protocol['evaluation_budget']
            compliant=(trace['actor_seconds']<=budget['max_task_seconds'] and trace['attempted_calls']<=budget['max_model_calls']
                and trace['generated_tokens']<=budget['max_total_generated_tokens']
                and all(s['generation']['generated_tokens']<=budget['max_generated_tokens_per_call'] and s['generation']['input_tokens']<=budget['max_input_tokens'] for s in trace['steps']))
            success=expected_grade['passed'] and compliant
            flags={'run_identity':trace['id']==task['id'] and trace['arm']==item['arm'] and trace['split']=='development'
                         and trace['training_export_allowed'] is False,
                   'protocol_identity':trace['protocol_sha256']==digest(json.dumps(protocol,sort_keys=True).encode()),
                   'original_source':trace['original_source_sha256']=={n:before[task['id']][n] for n in task['editable_files']},
                   'recorded_grade':expected_grade==trace['grade']==result['grade'],
                   'budget_score':compliant==trace['actor_budget_compliant'] and success==trace['success_within_budget']==result['success_within_budget'],
                   'result_in_controller':controller_map.get(item['run_id'])==result,
                   'trace_hash':result['trajectory_sha256']==sha256(path),
                   'request_replay':replayed['passed']}
            row.update(checks=flags,audit_passed=all(flags.values()),success_within_budget=bool(success),
                       grader_passed=expected_grade['passed'],grade_reason=expected_grade['reason'],
                       actor_seconds=trace['actor_seconds'],submitted=trace['submitted'],terminal_reason=trace['terminal_reason'],
                       first_edit_call=result['first_edit_call'],invalid_actions=result['invalid_actions'],
                       rejected_unexecuted_requests=result['rejected_unexecuted_requests'])
        except Exception as error:
            row.update(reason='audit_error',error_type=type(error).__name__)
            (audit_private/(item['run_id']+'-error.log')).write_text(traceback.format_exc())
        public_rows.append(row)
    unchanged=all((root/relative(n)).is_file() and sha256(root/relative(n))==h for n,h in protocol['frozen_files'].items())
    try:expected_environment=json.loads((root/relative(protocol['environment_path'])).read_text())
    except (OSError,ValueError):expected_environment=None
    environment_checks=[]
    for entry in controller.get('backend_lifecycles',[]):
        path=run_dir/'private/backends'/f"{entry['index']:04d}-{entry['role']}-environment.json"
        try:matched=expected_environment is not None and json.loads(path.read_text())==expected_environment
        except (OSError,ValueError):matched=False
        environment_checks.append(matched)
    checks={'all_runs_complete':controller['complete'] and len(controller.get('runs',[]))==len(protocol['schedule']),
            'all_run_audits_passed':all(r['audit_passed'] for r in public_rows),'frozen_inputs_unchanged':unchanged,
            'controller_protocol_identity':controller.get('protocol_sha256')==sha256(protocol_path),
            'supervisor_completed_without_timeout':supervisor['complete'] and not supervisor['timed_out'] and supervisor['worker_returncode']==0,
            'same_registered_model_interpreter':bool(environment_checks) and all(environment_checks)}
    result={'schema_version':1,'protocol_sha256':sha256(protocol_path),'checks':checks,'passed':all(checks.values()),
            'rows':public_rows,'model_generations':0,'optimizer_updates':0,'test_commands_reexecuted':0,
            'grader_input_and_recorded_output_checked':True,'final_test_opened':False,'weight_publication_allowed':False}
    write_json(out,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','protocol','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();value=run(a.root,a.protocol,a.out);print(json.dumps({'passed':value['passed'],'checks':value['checks']}))
