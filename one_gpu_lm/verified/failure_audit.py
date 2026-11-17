"""Replay all collection requests and export only verified corrective targets."""
import argparse
import json
from pathlib import Path
import traceback
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.reasoning import action_text, settings
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.common.receipts import sha256, write_json
from .failure_continuation import behavior, load_snapshot, initial_messages
from .recovery_state import canonical
from .failure_replay import replay
from .evaluation_audit import recorded_grade
from .evaluation_rollout import normalization
from .guard_audit import target_is_executed
from .submission import SubmissionGuard, SUBMIT
from .tasks import prepare, relative, digest


def encode_correction(directory,trace,task,p,profile,tokenizer,prefix,selected,student_profile=None):
    chain=behavior(trace,task);records=[];excluded=[];errors=[];receipts=[]
    if not chain['qualified_behavior']:
        return dict(chain,qualified=False,encoded_actions=[],excluded_calls=[],encoding_errors=[]),records
    start=chain['edit_calls'][-1]
    required_edit=prefix['messages'][2*prefix['failed_edit_call']]['content']
    required_feedback=initial_messages(prefix,p,trace['branch'])[-1]['content']
    for step in trace['steps'][start-1:]:
        if not target_is_executed(step):
            excluded.append({'call':step['call'],'reason':'malformed_or_unexecuted'});continue
        try:
            payload=json.loads((Path(directory)/f"request-{step['call']:03d}.json").read_text());captured={}
            def encode(messages):
                captured['messages']=messages
                return encode_prefix(tokenizer,messages,settings(profile,payload)[0],p['agent_policy']['format'])[0]
            tokens,context=prepare_context(payload['messages'],encode,payload['max_input_tokens'],p['agent_policy']['context'])
            if (digest(json.dumps(tokens,separators=(',',':')).encode())!=step['generation']['input_tokens_sha256']
                    or context!=step['generation']['context']):raise ValueError('Actual continuation input does not replay')
            if step['call']==start:
                contents=[m['content'] for m in captured['messages']]
                if required_edit not in contents or required_feedback not in contents:
                    raise ValueError('Historical failed edit or confirmed failure absent from actual repair input')
            kind=('edit' if step.get('source_after')!=step['source_before'] else 'verify' if step['command']==task['regression_command']
                  else 'submit' if step['command']==SUBMIT else 'other')
            row={'id':task['id']+':'+trace['branch']+':'+str(step['call']),'messages':captured['messages'],'target':action_text(step['generation'],p),
                 'provenance':{'split':'train','task_id':task['id'],'repository':task['repository'],'arm':trace['branch'],
                    'call':step['call'],'kind':kind,'producer_kind':'model','model_role':'base','executed':True,
                    'state_id':selected['state_id'],'admitted_prefix_sha256':selected['snapshot']['prefix.json'],
                    'trajectory_sha256':sha256(Path(directory)/'trajectory.json'),'request_sha256':step['request_sha256'],
                    'historical_failed_edit_call':prefix['failed_edit_call'],'historical_failure_call':prefix['failure_call'],
                    'target_start_call':start}}
            export_profile=student_profile or profile
            student_tokens,_=encode_prefix(tokenizer,captured['messages'],export_profile['chat_template_kwargs'],p['agent_policy']['format']) if student_profile else (tokens,{})
            encoded=encode_training_example(row,tokenizer,export_profile)
            if (encoded['ids'][:len(student_tokens)]!=student_tokens or len(encoded['ids'])>p['encoding_screen_max_tokens'] or encoded['supervised_tokens']<=0):
                raise ValueError('Untruncated corrective target does not fit')
            receipt={'call':step['call'],'kind':kind,'tokens':len(encoded['ids']),'supervised_tokens':encoded['supervised_tokens']}
            if student_profile:
                receipt.update(export_mode='action_only_cross_mode',teacher_input_sha256=step['generation']['input_tokens_sha256'],
                    student_input_sha256=digest(json.dumps(student_tokens,separators=(',',':')).encode()),
                    student_profile_sha256=p['reasoning_comparison']['student_profile_sha256'],teacher_mode=payload['generation_mode'])
                row['provenance']['target_transform']='verified_final_action_only_no_thought_labels'
            row['encoding_receipt']=receipt;records.append(row);receipts.append(receipt)
        except (ValueError,KeyError) as error:errors.append({'call':step['call'],'error_type':type(error).__name__})
    qualified=not errors and {'edit','verify','submit'}<={r['kind'] for r in receipts}
    return dict(chain,qualified=qualified,target_start_call=start,encoded_actions=receipts,excluded_calls=excluded,encoding_errors=errors),records


def physical_attempts(out,schedule):
    rows=[]
    for item in schedule:
        directory=Path(out)/'private/tasks'/item['run_id'];path=directory/'trajectory.json'
        if not path.exists(): continue
        t=json.loads(path.read_text());requests=list(directory.glob('request-*.json'));steps=t['steps']
        attempted=max(t['attempted_calls'],len(requests))
        rows.append({'run_id':item['run_id'],'attempted_calls':attempted,
                     'generated_tokens':sum(s['generation']['generated_tokens'] for s in steps),
                     'input_tokens':sum(s['generation']['input_tokens'] for s in steps),'actor_seconds':t['actor_seconds'],
                     'unknown_generation_cost':bool(t.get('unknown_inflight_generation_cost')) or attempted!=len(steps) or attempted!=len(requests),
                     'trajectory_sha256':sha256(path)})
    return rows


def run(root,protocol_path,out):
    from transformers import AutoTokenizer
    root,protocol_path,out=Path(root),Path(protocol_path),Path(out)
    if out.exists(): raise ValueError('Preserve previous audit')
    p=json.loads(protocol_path.read_text());run_dir=root/relative(p['run_directory']);c=json.loads((run_dir/'controller.json').read_text())
    supervisor=json.loads((root/relative(p['supervisor_receipt'])).read_text())
    if not c.get('backend_stopped') or not supervisor['complete']: raise ValueError('Owned processes must stop first')
    data=root/relative(p['data_directory']);tasks={t['id']:t for t in json.loads((data/'tasks.json').read_text())['tasks']}
    sources={s['repository']:s for s in json.loads((data/'sources.json').read_text())['sources']}
    selected_by_task={r['task_id']:r for r in p['selected_states']}
    profile=json.loads((root/relative(p['evaluation_profile_path'])).read_text());runtime=json.loads((root/'private/parent-runtime.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False);norm=normalization(root)
    student_profile=None
    if p.get('reasoning_comparison'):
        student_path=root/relative(p['reasoning_comparison']['student_profile_path'])
        if sha256(student_path)!=p['reasoning_comparison']['student_profile_sha256']:raise ValueError('Student profile changed')
        student_profile=json.loads(student_path.read_text())
    reconstruction=run_dir/'private/audit-reconstruction';reconstruction.mkdir()
    export=run_dir/'private/correction-export';export.mkdir();by_id={r['run_id']:r for r in c['runs']};rows=[];records=[]
    for item in p['schedule']:
        task=tasks[item['task_id']];source=sources[task['repository']];directory=run_dir/'private/tasks'/item['run_id']
        selected=selected_by_task[task['id']]
        row=dict(item,family=task['family'],state_id=selected['state_id'],previously_qualified_027=task['id'] in p['known_corrective_task_ids'],complete=False,audit_passed=False,qualified=False)
        if item['run_id'] not in by_id: rows.append(row);continue
        try:
            before=prepare(data/'snapshots'/task['repository'].replace('/','--'),reconstruction/'broken'/item['run_id'],source,task)
            t=json.loads((directory/'trajectory.json').read_text());value=json.loads((directory/'result.json').read_text())
            snapshot,prefix,manifest=load_snapshot(root,selected)
            replayed=replay(directory,t,task,p,profile,tokenizer,norm,prefix)
            grade=recorded_grade(root,data,directory,task,source,before,reconstruction/'graders'/item['run_id'])
            b=p['evaluation_budget'];compliant=(t['actor_seconds']<=b['max_task_seconds'] and t['attempted_calls']<=b['max_model_calls']
                and t['generated_tokens']<=b['max_total_generated_tokens'] and all(s['generation']['generated_tokens']<=b['max_generated_tokens_per_call'] and s['generation']['input_tokens']<=b['max_input_tokens'] for s in t['steps']))
            final=SubmissionGuard.source_state(directory/'workspace',task)
            verified=replayed['reconstructed_verified_source'] is not None and replayed['reconstructed_verified_source']==final
            submitted=bool(t['steps'] and t['steps'][-1].get('command')==SUBMIT and t['steps'][-1].get('observation',{}).get('returncode')==0 and t['steps'][-1].get('observation',{}).get('executed') is True)
            encoded,raw=encode_correction(directory,t,task,p,profile,tokenizer,prefix,selected,student_profile)
            first_evidence=json.loads((directory/'source-states'/(t['steps'][0]['source_evidence_before']+'.json')).read_text()) if t['steps'] else None
            checks={'request_replay':replayed['passed'],'recorded_grader':grade==t['grade'],
                'identity':task['split']==t['split']=='training' and t['id']==task['id'] and t['branch']==item['arm'] and t['protocol_sha256']==digest(json.dumps(p,sort_keys=True).encode()),
                'original_source':t['original_source_sha256']=={n:before[n] for n in task['editable_files']},
                'original_inventory':json.loads((directory/'before.json').read_text())==before,
                'final_source':final==t['final_source_sha256']==replayed['reconstructed_final_source'],
                'score':compliant==t['actor_budget_compliant']==value['actor_budget_compliant'] and bool(grade['passed'] and compliant)==t['success_within_budget']==value['success_within_budget'],
                'verified_submission':verified==t['agent_verified_final_source']==value['agent_verified_final_source'] and submitted==t['submitted']==value['submitted'] and bool(grade['passed'] and compliant and verified and submitted)==t['eligible_verified_trajectory']==value['eligible_verified_trajectory'],
                'costs':all(value[k]==t[k] for k in ['attempted_calls','generated_tokens','actor_seconds']) and value['input_tokens']==replayed['known_input_tokens'],
                'recorded_result':value==by_id[item['run_id']] and value['trajectory_sha256']==sha256(directory/'trajectory.json'),
                'behavior_rule':value['correction_behavior']==behavior(t,task),
                'restored_state':json.loads((directory/'restored-state.json').read_text())=={'state_id':selected['state_id'],'snapshot':selected['snapshot'],'manifest_sha256':digest(canonical(manifest))},
                'initial_workspace_contents':first_evidence is not None and first_evidence['workspace_inventory']=={n:v['sha256'] for n,v in manifest.items() if v['kind']=='file'},
                'admitted_source':t['state_id']==selected['state_id'] and t['failed_source_sha256']==prefix['failed_source_sha256'] and t['original_source_sha256']==prefix['original_source_sha256'],
                'historical_cost':t['historical_prefix_cost']=={k:prefix[k] for k in ['historical_calls','historical_generated_tokens','historical_input_tokens']}}
            row.update(result=value,encoding=encoded,historical_prefix_cost=t['historical_prefix_cost'],checks=checks,complete=True,audit_passed=all(checks.values()))
            if p.get('reasoning_comparison'):
                row['reasoning_usage']={k:sum(s['generation']['separation'][k] for s in t['steps']) for k in ['thought_tokens','action_tokens','unclassified_tokens']}
                row['reasoning_usage']['boundary_failures']=sum(not s['generation']['separation']['valid'] for s in t['steps'])
                row['reasoning_usage']['mode']=p['reasoning_comparison']['arms'][item['arm']]
            row['qualified']=bool(encoded['qualified'] and row['audit_passed'])
            if row['qualified'] and item['arm']=='correction': records.extend(raw)
        except Exception as error:
            row['error_type']=type(error).__name__; (reconstruction/(item['run_id']+'-error.log')).write_text(traceback.format_exc())
        rows.append(row)
    attempts=physical_attempts(run_dir,p['schedule']);environments=[json.loads((run_dir/'private/backends'/f"{e['index']:04d}-{e['role']}-environment.json").read_text())==json.loads((root/relative(p['environment_path'])).read_text()) for e in c['backend_lifecycles']]
    checks={'all_runs_audited':c['complete'] and len(rows)==len(p['schedule']) and all(r['complete'] and r['audit_passed'] for r in rows),
        'all_generation_cost_known':all(not a['unknown_generation_cost'] for a in attempts),
        'global_caps':sum(a['attempted_calls'] for a in attempts)<=p['global_caps']['model_calls'] and sum(a['generated_tokens'] for a in attempts)<=p['global_caps']['generated_tokens'],
        'physical_cost_matches':sum(a['attempted_calls'] for a in attempts)==sum(r['attempted_calls'] for r in c['runs']) and sum(a['generated_tokens'] for a in attempts)==sum(r['generated_tokens'] for r in c['runs']),
        'no_unregistered_task_directory':{d.name for d in (run_dir/'private/tasks').iterdir()}<={i['run_id'] for i in p['schedule']},
        'frozen_inputs_unchanged':all(sha256(root/relative(n))==h for n,h in p['frozen_files'].items()),
        'same_model_interpreter':bool(environments) and all(environments),
        'supervisor_complete':supervisor['complete'] and not supervisor['timed_out'] and supervisor['worker_returncode']==0,
        'controller_protocol_identity':c['protocol_sha256']==sha256(protocol_path)}
    (export/'data.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    manifest={'qualified_unique_tasks':len({r['provenance']['task_id'] for r in records}),'action_records':len(records),
              'supervised_tokens':sum(r['encoding_receipt']['supervised_tokens'] for r in records),'data_sha256':sha256(export/'data.jsonl'),
              'failed_edits_used_as_positive_targets':False,'historical_actions_used_as_positive_targets':False,'failed_edits_retained_as_context':True,'raw_data_public':False}
    write_json(export/'manifest.json',manifest)
    result={'schema_version':1,'protocol_sha256':sha256(protocol_path),'passed':all(checks.values()),'checks':checks,'rows':rows,
            'attempts':attempts,'private_export_summary':manifest,'model_generations':0,'optimizer_updates':0,'test_commands_reexecuted':0,'final_test_opened':False,'weights_public':False}
    write_json(out,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['root','protocol','out']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();v=run(args.root,args.protocol,args.out);print(json.dumps({'passed':v['passed'],'checks':v['checks']}))
