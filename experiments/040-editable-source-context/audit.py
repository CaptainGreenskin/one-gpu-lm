"""Replay the source-context comparison without issuing new model requests."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_tasks import prepare
from one_gpu_lm.verified.rollout import normalization

from study import OUT, REPLAYER, inputs, load, read


def audit(root):
    _,context,prompts = inputs(root)
    from analysis import analyze
    from transformers import AutoTokenizer

    out = root/OUT
    target = out/'audit.json'
    if target.exists():
        raise FileExistsError('Preserve the first audit')
    controller = out/'controller.json'
    state = read(controller)
    if (state['stage'] != 'completed' or not state['complete'] or not state['backend_stopped']
            or not state['frozen_inputs_unchanged'] or state['optimizer_updates'] or state['final_test_opened']
            or state['weights_public'] or state['training_export_allowed']
            or state['registration_sha256'] != sha256(context.plan_path) or not state['telemetry']['thread_stopped']
            or state['schedule_sha256'] != context.plan['schedule_sha256']
            or not state['started_epoch'] < state['finished_epoch'] <= state['started_epoch']+context.plan['pipeline_limit_seconds']
            or [r['run_id'] for r in state['runs']] != [r['run_id'] for r in context.order]):
        raise ValueError('Every frozen source-context pair is required')

    backends = state['backend_lifecycles']
    if len(backends) != 1 or backends[0]['role'] != 'base' or backends[0]['index'] != 1:
        raise ValueError('Require a single common base model load')
    backend = backends[0]
    location = out/'private/backends/0001-base'
    served = read(location/'served-artifact.json')
    protocol = context.protocol
    if (served['model_id'] != protocol['model_id'] or served['revision'] != protocol['revision']
            or served['adapter'] or served['adapter_sha256'] or served['request_journal_version'] != 1
            or served['profile_sha256'] != protocol['evaluation_profile_sha256']
            or read(location.with_name('0001-base-environment.json')) != read(root/protocol['environment_path'])
            or not state['started_epoch'] <= backend['ready_epoch'] <= backend['stopped_epoch'] <= state['finished_epoch']):
        raise ValueError('Served model or runtime does not match the fixed base')

    sys.path.insert(0,str(root/REPLAYER.parent))
    replay = load('source_context_audited_row',root/REPLAYER)
    tokenizer = AutoTokenizer.from_pretrained(context.runtime['model'],local_files_only=True,trust_remote_code=False)
    profile = read(root/protocol['evaluation_profile_path'])
    norm = normalization(root)
    journal = RequestJournal(location/'requests')
    proof = {r['task_number']:r for r in context.legacy_proof['rows']}
    rows,seen,original = [],set(),{}
    with tempfile.TemporaryDirectory(prefix='040-audit-',dir=out/'private') as temp:
        scratch = Path(temp)
        for planned,result in zip(context.order,state['runs']):
            n,arm,run_id = (planned[k] for k in ('task_number','arm','run_id'))
            task = prompts[run_id]
            source = context.sources[task['source_id']]
            if (result['task_number'] != n or result['contrast_arm'] != arm or result['id'] != task['id']
                    or result['block'] != planned['block'] or result['backend_index'] != 1):
                raise ValueError('Result is attached to the wrong scheduled contrast')
            directory = out/'private/tasks'/run_id
            trace = read(directory/'trajectory.json')
            if trace['backend'] != served:
                raise ValueError('Rollout used another model load')
            evidence = proof[n]
            if n not in original:
                before = prepare(context.data/'snapshots'/source['source_id'],scratch/f'original-{n:02d}',source,task)
                if fingerprint(sorted(before.items())) != evidence['workspace_sha256']:
                    raise ValueError('Original protected workspace changed')
                original[n] = before
            row,opening,requests = replay.audited_row(root,context.data,directory,result,task,source,protocol,
                                                      profile,tokenizer,norm,journal,scratch/run_id,original[n],backend)
            if (opening != evidence['input_sha256'][arm] or requests & seen
                    or trace['steps'][0]['generation']['input_tokens'] != evidence['input_tokens'][arm]):
                raise ValueError('Opening tokens or request accounting differ from preflight')
            seen.update(requests)
            row.update(repetition=0,multifile=evidence['multifile'],opening_input_tokens=evidence['input_tokens'][arm],
                       source_window_lines=evidence['window_lines'])
            rows.append(row)
    if seen != {path.name for path in journal.directory.iterdir()} or len(seen) != sum(r['attempted_calls'] for r in rows):
        raise ValueError('Model journal and audited actions differ')
    summary = {'schema_version':1,'passed':True,'protocol_sha256':sha256(context.plan_path),
               'controller_sha256':sha256(controller),'verified_requests':len(seen),'started_epoch':state['started_epoch'],
               'finished_epoch':state['finished_epoch'],'backend_loads':len(backends),'telemetry':state['telemetry'],
               'rows':rows,'analysis':analyze(rows,context.plan),'optimizer_updates':0,'final_test_opened':False,
               'weights_public':False}
    write_json(target,summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    result = audit(p.parse_args().root.resolve())
    print(json.dumps({'passed':result['passed'],'verified_requests':result['verified_requests'],
                      'followup_gate_passed':result['analysis']['development_gate_passed']}))
