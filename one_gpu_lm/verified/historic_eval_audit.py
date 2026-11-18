"""Replay every 032 variant with its own checkpoint-bound effective protocol."""
import argparse
import json
from pathlib import Path
import traceback
from one_gpu_lm.common.receipts import sha256,write_json
from .historic_replay import replay,recorded_grade
from .evaluation_rollout import normalization
from .historic_eval_design import effective_protocol,internal_arm,VARIANTS
from .tasks import relative,digest
from .historic_tasks import prepare
from .historic_outcomes import executed_completion


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
    sources={s['source_id']:s for s in json.loads((data/'sources.json').read_text())['sources']}
    norm=normalization(root);before={};audit_private=run_dir/'private/audit-reconstruction'
    if audit_private.exists():raise ValueError('Preserve an interrupted audit reconstruction')
    audit_private.mkdir(parents=True)
    for task in tasks.values():
        source=sources[task['source_id']];snapshot=data/'snapshots'/task['source_id']
        before[task['id']]=prepare(snapshot,audit_private/'broken'/task['id'],source,task)
    public_rows=[];controller_map={r['run_id']:r for r in controller.get('runs',[])}
    for item in protocol['schedule']:
        task=tasks[item['task_id']];effective=effective_protocol(protocol,item['variant']);arm=internal_arm(item['variant']);directory=run_dir/'private/tasks'/item['run_id'];path=directory/'trajectory.json'
        row={'id':task['id'],'repository':task['repository'],'family':task['family'],'arm':arm,'variant':item['variant'],'block':item['block'],
             'repetition':item['repetition'],'run_id':item['run_id'],'complete':False,'audit_passed':False}
        if not path.exists():row['reason']='not_run';public_rows.append(row);continue
        trace=json.loads(path.read_text())
        row.update(complete=trace['complete'],attempted_calls=trace['attempted_calls'],generated_tokens=trace['generated_tokens'],
                   seconds=trace.get('seconds',0),unknown_inflight_generation_cost=bool(trace.get('unknown_inflight_generation_cost',False) or trace['attempted_calls']!=len(trace['steps']) or len(list(directory.glob('request-*.json')))!=len(trace['steps'])),
                   trajectory_sha256=sha256(path))
        try:
            replayed=replay(directory,trace,task,effective,profile,tokenizer,norm)
            row.update(input_tokens=replayed['known_input_tokens'],request_replay=replayed)
            if not trace['complete']:
                row['reason']='partial_trajectory';public_rows.append(row);continue
            result=json.loads((directory/'result.json').read_text())
            expected_grade=recorded_grade(root,data,directory,task,sources[task['source_id']],before[task['id']],audit_private/'graders'/item['run_id'])
            budget=protocol['evaluation_budget']
            compliant=(trace['actor_seconds']<=budget['max_task_seconds'] and trace['attempted_calls']<=budget['max_model_calls']
                and trace['generated_tokens']<=budget['max_total_generated_tokens']
                and all(s['generation']['generated_tokens']<=budget['max_generated_tokens_per_call'] and s['generation']['input_tokens']<=budget['max_input_tokens'] for s in trace['steps']))
            success=expected_grade['passed'] and compliant
            final_source={n:sha256(directory/'workspace'/n) for n in task['editable_files']
                          if (directory/'workspace'/n).is_file() and not (directory/'workspace'/n).is_symlink()}
            completion=executed_completion(trace,task,final_source)
            flags={'run_identity':trace['id']==task['id'] and trace['arm']==arm and result['variant']==item['variant'] and result['repetition']==item['repetition'] and result['block']==item['block'] and trace['split']=='development'
                         and trace['training_export_allowed'] is False,
                   'protocol_identity':trace['protocol_sha256']==digest(json.dumps(effective,sort_keys=True).encode()),
                   'original_source':trace['original_source_sha256']=={n:before[task['id']][n] for n in task['editable_files']},
                   'recorded_grade':expected_grade==trace['grade']==result['grade'],
                   'executed_completion':all(trace[k]==result[k]==v for k,v in completion.items()),
                   'final_source_identity':trace['final_source_sha256']==final_source==result['final_source_sha256'],
                   'completion_score':(success and completion['submitted'])==trace['complete_success_within_budget']==result['complete_success_within_budget'],
                   'budget_score':compliant==trace['actor_budget_compliant'] and success==trace['success_within_budget']==result['success_within_budget'],
                   'result_in_controller':controller_map.get(item['run_id'])==result,
                   'trace_hash':result['trajectory_sha256']==sha256(path),
                   'request_replay':replayed['passed'],
                   'complete_result_costs':all(result[k]==trace[k] for k in ['attempted_calls','generated_tokens','actor_seconds','seconds','submitted','terminal_reason']) and result['input_tokens']==replayed['known_input_tokens'],
                   'behavior_counts':result['invalid_actions']==sum(not s['action_format_valid'] for s in trace['steps']) and result['first_edit_call']==next((s['call'] for s in trace['steps'] if s.get('source_after') and s['source_after']!=s['source_before']),None),
                   'no_runtime_intervention':result['rejected_unexecuted_requests']==0 and trace['runtime_variant']=='current'}
            row.update(checks=flags,audit_passed=all(flags.values()),success_within_budget=bool(success),complete_success_within_budget=bool(success and trace['submitted']),
                       grader_passed=expected_grade['passed'],grade_reason=expected_grade['reason'],
                       actor_seconds=trace['actor_seconds'],submitted=completion['submitted'],
                       agent_verified_final_source=completion['agent_verified_final_source'],terminal_reason=trace['terminal_reason'],
                       first_edit_call=result['first_edit_call'],invalid_actions=result['invalid_actions'],
                       rejected_unexecuted_requests=result['rejected_unexecuted_requests'])
        except Exception as error:
            row.update(reason='audit_error',error_type=type(error).__name__)
            (audit_private/(item['run_id']+'-error.log')).write_text(traceback.format_exc())
        public_rows.append(row)
    physical=[]
    for item in protocol['schedule']:
        directory=run_dir/'private/tasks'/item['run_id'];path=directory/'trajectory.json'
        if not path.exists():continue
        t=json.loads(path.read_text());steps=t['steps'];requests=list(directory.glob('request-*.json'))
        physical.append({'run_id':item['run_id'],'attempted_calls':max(t['attempted_calls'],len(requests)),
          'generated_tokens':sum(s['generation']['generated_tokens'] for s in steps),
          'input_tokens':sum(s['generation']['input_tokens'] for s in steps),
          'unknown_generation_cost':bool(t.get('unknown_inflight_generation_cost')) or t['attempted_calls']!=len(steps) or len(requests)!=len(steps)})
    unchanged=all((root/relative(n)).is_file() and sha256(root/relative(n))==h for n,h in protocol['frozen_files'].items())
    try:expected_environment=json.loads((root/relative(protocol['environment_path'])).read_text())
    except (OSError,ValueError):expected_environment=None
    environment_checks=[]
    for entry in controller.get('backend_lifecycles',[]):
        path=run_dir/'private/backends'/f"{entry['index']:04d}-{entry['role']}-environment.json"
        try:matched=expected_environment is not None and json.loads(path.read_text())==expected_environment
        except (OSError,ValueError):matched=False
        variant=entry.get('variant');served=run_dir/'private/backends'/f"{entry['index']:04d}-{entry['role']}"/'served-artifact.json'
        if variant not in VARIANTS or not served.is_file():matched=False
        else:
            artifact=json.loads(served.read_text());effective=effective_protocol(protocol,variant)
            matched=bool(matched and artifact['adapter_sha256']==effective['adapter_sha256']
                and artifact['profile_sha256']==protocol['evaluation_profile_sha256']
                and entry['role']==('base' if variant=='base' else 'student') and 'stopped_epoch' in entry)
        environment_checks.append(matched)
    expected_loads=[]
    for item in protocol['schedule']:
        if not expected_loads or expected_loads[-1]!=item['variant']:expected_loads.append(item['variant'])
    checks={'all_runs_complete':controller['complete'] and len(controller.get('runs',[]))==len(protocol['schedule']),
            'all_run_audits_passed':all(r['audit_passed'] for r in public_rows),'frozen_inputs_unchanged':unchanged,
            'controller_protocol_identity':controller.get('protocol_sha256')==sha256(protocol_path),
            'supervisor_protocol_identity':supervisor.get('protocol_sha256')==sha256(protocol_path),
            'development_only_no_updates':controller.get('optimizer_updates_during_evaluation')==0
                and controller.get('final_test_opened') is False and controller.get('weights_public') is False,
            'supervisor_completed_without_timeout':supervisor['complete'] and not supervisor['timed_out'] and supervisor['worker_returncode']==0,
            'same_registered_model_interpreter':bool(environment_checks) and all(environment_checks),
            'physical_costs_accounted':not any(x['unknown_generation_cost'] for x in physical)
                and sum(x['attempted_calls'] for x in physical)==sum(r.get('attempted_calls',0) for r in controller.get('runs',[]))
                and sum(x['generated_tokens'] for x in physical)==sum(r.get('generated_tokens',0) for r in controller.get('runs',[])),
            'no_unregistered_task_directory':({d.name for d in (run_dir/'private/tasks').iterdir()} if (run_dir/'private/tasks').exists() else set())<={r['run_id'] for r in protocol['schedule']},
            'controller_order_and_uniqueness':[r['run_id'] for r in controller.get('runs',[])]==[r['run_id'] for r in protocol['schedule']],
            'at_most_54_model_loads':len(controller.get('backend_lifecycles',[]))<=54,
            'exact_registered_checkpoint_load_order':[r.get('variant') for r in controller.get('backend_lifecycles',[])]==expected_loads}
    result={'schema_version':1,'protocol_sha256':sha256(protocol_path),'checks':checks,'passed':all(checks.values()),
            'rows':public_rows,'physical_attempts':physical,'model_generations':0,'optimizer_updates':0,'test_commands_reexecuted':0,
            'grader_input_and_recorded_output_checked':True,'final_test_opened':False,'weight_publication_allowed':False}
    write_json(out,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','protocol','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();value=run(a.root,a.protocol,a.out);print(json.dumps({'passed':value['passed'],'checks':value['checks']}))
