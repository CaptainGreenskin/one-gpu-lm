"""Audit input histories, generation costs, and recorded regrading for 034."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from transformers import AutoTokenizer

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.first_action_design import analyze
from one_gpu_lm.verified.historic_outcomes import executed_completion
from one_gpu_lm.verified.historic_replay import recorded_grade, replay
from one_gpu_lm.verified.historic_tasks import prepare
from one_gpu_lm.verified.resilient_eval_audit import audit_task
from one_gpu_lm.verified.rollout import normalization
from one_gpu_lm.verified.tasks import digest

from evaluate import EXP, OUTPUT, PRIVATE, effective, inputs, read


def run(root, out):
    if out.exists():
        raise ValueError('Preserve first development audit')
    plan,parent,data,tasks,sources,order=inputs(root)
    files=root/OUTPUT
    controller=read(files/'controller.json')
    if (not controller['complete'] or not controller['backend_stopped'] or not controller['frozen_inputs_unchanged']
            or controller['registration_sha256']!=sha256(root/EXP/'evaluation-plan.json')
            or controller['training_integrity_sha256']!=sha256(root/PRIVATE/'training-integrity.json')
            or controller['final_test_opened'] or controller['optimizer_updates']!=0
            or [r['run_id'] for r in controller['runs']]!=[r['run_id'] for r in order]):
        raise ValueError('Complete, exact frozen development sweep required')
    runtime=read(root/'private/parent-runtime.json')
    profile=read(root/parent['evaluation_profile_path'])
    tokenizer=AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False)
    norm=normalization(root)
    training=read(root/PRIVATE/'training-integrity.json')
    protocols={a:effective(root,parent,plan,training,a) for a in plan['arms']}
    journals={}
    seen=defaultdict(set)
    backends=controller['backend_lifecycles']
    loads=[item['arm'] for i,item in enumerate(order) if i==0 or item['arm']!=order[i-1]['arm']]
    if [b['arm'] for b in backends]!=loads:
        raise ValueError('Checkpoint loading order differs from registered order')
    environment=read(root/parent['environment_path'])
    for b in backends:
        if b['role']!='student' or not b['started_epoch'] <= b['ready_epoch'] <= b['stopped_epoch']:
            raise ValueError('Incomplete backend lifecycle')
        directory=files/'private/backends'/f"{b['index']:04d}-student"
        health=read(directory/'served-artifact.json')
        if (read(directory.with_name(directory.name+'-environment.json'))!=environment
                or health['adapter_sha256']!=protocols[b['arm']]['adapter_sha256']
                or health['profile_sha256']!=parent['evaluation_profile_sha256']):
            raise ValueError('Unexpected runtime or loaded checkpoint')
        journals[b['index']]=RequestJournal(directory/'requests')
    admission={r['id']:r for r in read(data/'admission.json')['rows']}
    task_map={t['id']:t for t in tasks}
    reconstruction=files/'private/audit'
    reconstruction.mkdir()
    broken={t['id']:prepare(data/'snapshots'/t['source_id'],reconstruction/'broken'/t['id'],sources[t['source_id']],t)
            for t in tasks}
    rows=[]
    budget=plan['evaluation_budget']
    for item, result in zip(order,controller['runs']):
        task=task_map[item['task_id']]; arm=item['arm']; run_id=item['run_id']; idx=result['backend_index']
        directory=files/'private/tasks'/run_id
        trace_path=directory/'trajectory.json'
        trace=read(trace_path)
        source=sources[task['source_id']]
        selected=protocols[arm]
        replayed=replay(directory,trace,task,selected,profile,tokenizer,norm)
        requests=audit_task(directory,trace,result,journals[idx])
        if seen[idx]&requests:
            raise ValueError('Duplicate completed generation across evaluation tasks')
        seen[idx].update(requests)
        clean=recorded_grade(root,data,directory,task,source,broken[task['id']],reconstruction/'graded'/run_id)
        submitted=executed_completion(trace,task,trace['final_source_sha256'])
        file_hashes={n:sha256(directory/'workspace'/n) for n in task['editable_files']
                     if (directory/'workspace'/n).is_file() and not (directory/'workspace'/n).is_symlink()}
        within=(trace['actor_seconds']<=budget['max_task_seconds'] and trace['attempted_calls']<=budget['max_model_calls']
                and trace['generated_tokens']<=budget['max_total_generated_tokens'])
        success=clean['passed'] and within and submitted['submitted']
        first_edit=next((s['call'] for s in trace['steps'] if s.get('source_after') and s['source_before']!=s['source_after']),None)
        if (read(directory/'result.json')!=result or result['contrast_arm']!=arm or result['task_number']!=item['task_number']
                or trace['id']!=task['id'] or trace['split']!='development' or trace['training_export_allowed']
                or trace['protocol_sha256']!=digest(json.dumps(selected,sort_keys=True).encode())
                or not trace['complete'] or not replayed['passed'] or not trace['steps']
                or result['trajectory_sha256']!=sha256(trace_path) or backends[idx-1]['arm']!=arm
                or clean!=trace['grade'] or clean!=result['grade'] or within!=trace['actor_budget_compliant']
                or file_hashes!=trace['final_source_sha256'] or file_hashes!=result['final_source_sha256']
                or any(trace[k]!=v or result[k]!=v for k,v in submitted.items())
                or result['first_edit_call']!=first_edit or success!=result['complete_success_within_budget']):
            raise ValueError(f'Replay, grade or accounting differs on {run_id}')
        first=trace['steps'][0]
        rows.append({'task_number':item['task_number'],'arm':arm,'family':task['family'],
                     'multifile':admission[task['id']]['multifile'],'audit_passed':True,
                     'completed_repair':success,'correct_without_submission':bool(clean['passed'] and within and not submitted['submitted']),
                     'first_exact_regression':'observation' in first and first.get('command')==task['regression_command'],
                     'first_edit_call':first_edit,'no_edit':first_edit is None,
                     'first_input_sha256':first['generation']['input_tokens_sha256'],'attempted_calls':trace['attempted_calls'],
                     'input_tokens':replayed['known_input_tokens'],'generated_tokens':trace['generated_tokens'],
                     'actor_seconds':trace['actor_seconds'],'timeout_recovered':result['timeout_call'] is not None})
    for i,journal in journals.items():
        existing={p.name for p in journal.directory.iterdir() if p.is_dir()}
        if existing!=seen[i]:
            raise ValueError('Unregistered journal entries or missing durable generation receipts')
    result={'passed':True,'registration_sha256':controller['registration_sha256'],
            'controller_sha256':sha256(files/'controller.json'),'new_model_calls':0,'tests_reexecuted':0,
            'verified_requests':sum(map(len,seen.values())),'backend_loads':len(backends),
            'started_epoch':controller['started_epoch'],'finished_epoch':controller['finished_epoch'],
            'telemetry':controller['telemetry'],'rows':rows, 'analysis':analyze(rows,tasks),'final_test_opened':False}
    write_json(out,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=run(a.root.resolve(),a.out.resolve())
    print(json.dumps({'passed':r['passed'],'verified_requests':r['verified_requests'],'paired_tasks':r['analysis']['paired_tasks']}))
