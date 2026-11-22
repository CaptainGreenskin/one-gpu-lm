"""Collect first-action forks and audit their independent outcomes before export."""

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

from prepare import DIR, OUTPUT, RUNTIME, read, require, source_cases
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.runstate import atomic_json, gpu_lock_path, identity_digest, single_writer, utc
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.models.hf_runtime import Runtime
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.serving.client import request
from one_gpu_lm.serving.journaled_backend import make_server
from one_gpu_lm.serving.receipt_client import request_identity
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.continuation_training_data import BASELINE, SECRET_SHAPES
from one_gpu_lm.verified.correction_rollout import advance, finish
from one_gpu_lm.verified.decision_state_audit import check_arm, packed_history
from one_gpu_lm.verified.decision_state_collection import initialize
from one_gpu_lm.verified.first_edit_continuation import message_hash
from one_gpu_lm.verified.guard_audit import target_is_executed
from one_gpu_lm.verified.rollout import normalization
from one_gpu_lm.verified.runtime_identity import verify
from one_gpu_lm.verified.sandbox import Sandbox
from one_gpu_lm.verified.tasks import relative


def frozen(root, plan):
    return (all(sha256(root/relative(name)) == checksum for name,checksum in plan['frozen_files'].items())
            and all(sha256(Path(name)) == checksum for name,checksum in plan['runtime_inputs'].items()))


def registered(args):
    plan, publication = read(args.plan), read(args.publication)
    require(publication['protocol_sha256'] == sha256(args.plan) and publication['anonymous_files_verified'] is True
            and publication['ci_conclusion'] == 'success' and plan['training_only'] is True
            and plan['final_test_opened'] is False and plan['output'] == str(OUTPUT) and frozen(args.root,plan),
            'Require the exact verified public registration and frozen training-only inputs')
    parent,design,_,data,cases = source_cases(args.root,Path(plan['collection']))
    require(plan['design_sha256'] == sha256(DIR/'design.json') and list(cases)==[r['task_id'] for r in plan['rows']],
            'Restored training cohort differs from the registration')
    profile = Path(plan['deployment_inputs']['evaluation_profile'])
    common = dict(model_id=design['model_id'],revision=design['base_revision'],
                  evaluation_profile_sha256=sha256(profile),agent_policy=parent['agent_policy'])
    for number,row in enumerate(plan['rows'],1):
        case = cases[row['task_id']]
        require(row['ordinal']==number and row['prefix_sha256']==case.proof['prefix_sha256']
                and row['source_sha256']==case.prefix['decision_source_sha256']
                and row['messages_sha256']==message_hash(row['messages']), 'Frozen actor fork differs')
        proof = dict(case.proof,request_messages_sha256=row['messages_sha256'],
                     packed_messages_sha256=row['packed_messages_sha256'],arms={'control':row})
        cases[row['task_id']] = replace(case,messages=row['messages'],proof=proof,
                                       protocol=dict(common,evaluation_budget=row['budget']))
    return plan,design,data,cases


def schedule(rows):
    for row in rows:
        number = row['ordinal']
        order = ('greedy','sampled') if number % 2 else ('sampled','greedy')
        for arm in order:
            yield row,arm,f'{number:02d}-{arm}'


def seed_settings(row, design):
    options = design['sampling']
    return {'seed':options['seed_offset']+row['ordinal'],'temperature':options['temperature'],'top_p':options['top_p']}


def sampled_generate(runtime, payload, options):
    torch = runtime.torch
    device = runtime.model.device.index
    generate = runtime.model.generate

    def draw(*args, **kwargs):
        kwargs.update(do_sample=True,temperature=options['temperature'],top_p=options['top_p'],top_k=0)
        return generate(*args,**kwargs)

    with torch.random.fork_rng(devices=[device]), patch.object(runtime.model,'generate',draw):
        torch.random.default_generator.manual_seed(options['seed'])
        torch.cuda.default_generators[device].manual_seed(options['seed'])
        result = runtime.generate(payload)
    return dict(result,first_action_sampling=options)


def serve(args, *, inputs=None):
    plan,design,_,_ = registered(args) if inputs is None else inputs
    output = args.root/relative(plan['output'])/'private/backend'
    require(not output.exists() and not args.socket.exists(), 'Preserve the original backend evidence')
    output.mkdir(parents=True)
    files = plan['deployment_inputs']
    profile_path = Path(files['evaluation_profile'])
    profile = read(profile_path)
    model = Path(read(args.root/RUNTIME)['model'])
    write_json(output/'environment.json',verify(read(Path(files['environment']))))
    write_json(output/'verified-model.json',verify_model(model,Path(files['model_manifest']),profile))
    runtime = Runtime(model,profile,output)
    seed_map = {request_identity(run_id,1):seed_settings(row,design) for row,arm,run_id in schedule(plan['rows'])
                if arm=='sampled'}

    class ForkJournal(RequestJournal):
        def execute(self, request_id, payload, generate):
            settings = seed_map.get(request_id)
            if settings is not None:
                return super().execute(request_id,payload,lambda value: sampled_generate(runtime,value,settings))
            return super().execute(request_id,payload,generate)

    identity = {'ready':True,'model_id':profile['model_id'],'revision':profile['revision'],
                'profile_sha256':sha256(profile_path),'adapter':False,'adapter_sha256':{}}
    journal = ForkJournal(output/'requests')
    with make_server(args.socket,journal,runtime.generate,identity) as server:
        atomic_json(output/'ready.json',dict(identity,request_journal_version=1))
        server.serve_forever(poll_interval=0.5)


def gpu_processes():
    pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                                   text=True,timeout=10)
    return {int(pid) for pid in pids.split()}


def collect(args, *, inputs=None, entrypoint=None):
    plan,design,data,cases = registered(args) if inputs is None else inputs
    root, out = args.root,args.root/relative(plan['output'])
    collection = Path(plan['collection'])
    runner = Sandbox(read(collection/'private/task-runtime.json'))
    staged = list(schedule(plan['rows']))
    controller = out/'controller.json'
    with single_writer(gpu_lock_path()), TemporaryDirectory(prefix='action-fork-') as sockets:
        require(not out.exists() and not gpu_processes(), 'New collection requires an unused directory and the idle GPU')
        (out/'private').mkdir(parents=True)
        start = time.time()
        state = {'protocol_sha256':sha256(args.plan),'phase':'loading','started_utc':utc(),'started_epoch':start,
                 'deadline_epoch':start+design['pipeline_seconds'],'runs':[],'expected_rollouts':len(staged),
                 'training_only':True,'optimizer_updates':0,'final_test_opened':False,'backend_stopped':False,
                 'complete':False}
        atomic_json(controller,state)
        socket = Path(sockets)/'local.sock'
        telemetry = Telemetry(out/'telemetry.csv',1)
        telemetry.start()
        backend = None
        try:
            python = read(root/RUNTIME)['model_python']
            script = entrypoint if entrypoint is not None else DIR/'run.py'
            options = [python,'-u',str(script),'serve','--root',str(root),'--plan',str(args.plan),
                       '--publication',str(args.publication),'--socket',str(socket)]
            with (out/'private/backend.log').open('xb') as log:
                backend = subprocess.Popen(options,cwd=root,stdout=log,stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL,start_new_session=True)
            state['backend_pid'] = backend.pid
            atomic_json(controller,state)
            ready = out/'private/backend/ready.json'
            while not ready.exists():
                require(backend.poll() is None and time.time()<start+600,'The single frozen base did not become ready')
                time.sleep(.5)
            state['backend_ready_epoch'] = time.time()
            require(request(socket,{'operation':'health'})==read(ready),'Serving identity differs from the load receipt')
            journal = ready.parent/'requests'
            state['phase'] = 'collecting'
            for row,arm,name in staged:
                case = cases[row['task_id']]
                reserve = case.budget['max_task_seconds']+case.budget['max_tool_seconds']+design['recovery_seconds_per_request']
                require(backend.poll() is None and gpu_processes()=={backend.pid} and time.time()+reserve<state['deadline_epoch'],
                        'A competing process or expired budget prevents the next complete actor attempt')
                state['active_run'] = name
                atomic_json(controller,state)
                directory = out/'private/tasks'/name
                trace = initialize(directory,case,'control')
                advance(root,case.task,case.protocol,directory,trace,socket,runner,'control',journal_path=journal,
                        recovery_seconds=design['recovery_seconds_per_request'])
                result = finish(data,case.task,case.source,case.protocol,directory,trace,runner)
                result.update(run_id=name,arm='control',sampling_arm=arm,ordinal=row['ordinal'],
                              timeout_call=trace.get('timeout_call'),accounting_drain_seconds=trace.get('accounting_drain_seconds',0.0),
                              complete_success_within_budget=trace['eligible_verified_trajectory'])
                atomic_json(directory/'result.json',result)
                state['runs'].append(result)
                atomic_json(controller,state)
                print(json.dumps({'completed_rollouts':len(state['runs']),'expected_rollouts':len(staged)}),flush=True)
            state.update(phase='completed',active_run=None,complete=True)
        except BaseException as error:
            state.update(phase='failed',error_type=type(error).__name__)
            raise
        finally:
            try:
                if backend is not None and backend.poll() is None:
                    backend.terminate()
                    try:
                        backend.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        backend.kill()
                        backend.wait(timeout=8)
            finally:
                state.update(finished_epoch=time.time(),finished_utc=utc(),telemetry=telemetry.stop(),
                             backend_stopped=backend is None or backend.poll() is not None,
                             frozen_inputs_unchanged=frozen(root,plan))
                atomic_json(controller,state)


def compare_first(case, trajectories, scores, directories, tokenizer, profile, student):
    first = {arm:trace['steps'][0] for arm,trace in trajectories.items()}
    eligible = {arm:cell['verified_and_submitted'] for arm,cell in scores.items()}
    require(len({step['generation']['input_tokens_sha256'] for step in first.values()}) == 1
            and len({step['request_sha256'] for step in first.values()}) == 1
            and all(step['source_before']==case.prefix['decision_source_sha256'] for step in first.values()),
            'The two physical branches did not use identical inputs and restored sources')
    if len(set(eligible.values()))==1:
        return None,'same_outcome'
    if any(trace.get('timeout_call') is not None or trace['terminal_reason'].startswith('time_limit')
           or any(not step.get('protected_before') or not step.get('protected_after')
                  for step in trace['steps'] if 'observation' in step) for trace in trajectories.values()):
        return None,'time_or_protected_source'
    if not all(target_is_executed(step) for step in first.values()):
        return None,'first_action_not_executed'
    if first['greedy']['command']==first['sampled']['command']:
        return None,'same_first_command'
    chosen,rejected = sorted(eligible,key=eligible.get,reverse=True)
    messages = packed_history(directories[chosen],first[chosen],profile,tokenizer,case.protocol['agent_policy'])
    targets = [first[arm]['generation']['text'] for arm in (chosen,rejected)]
    try:
        encodings = [encode_training_example({'messages':messages,'target':text},tokenizer,student) for text in targets]
    except ValueError:
        return None,'template_incompatible'
    prefix = encodings[0]['prefix_tokens']
    if (any(len(entry['ids'])>student['training']['max_sequence_length'] or entry['supervised_tokens']>
            case.budget['max_generated_tokens_per_call'] for entry in encodings)
            or encodings[1]['prefix_tokens']!=prefix or encodings[0]['ids'][:prefix]!=encodings[1]['ids'][:prefix]
            or identity_digest(encodings[0]['ids'][:prefix])!=first[chosen]['generation']['input_tokens_sha256']
            or encodings[0]['ids']==encodings[1]['ids']):
        return None,'token_budget_or_shared_prefix'
    record = {'messages':messages,'chosen':targets[0],'rejected':targets[1],'kind':'early_edit',
              'provenance':{'task_id':case.task['id'],'split':'train','family':case.task['family'],
                            'prefix_sha256':case.proof['prefix_sha256'],'chosen_arm':chosen,'rejected_arm':rejected,
                            'fork_call':1,'chosen_request_sha256':first[chosen]['request_sha256'],
                            'rejected_request_sha256':first[rejected]['request_sha256'],
                            'remaining_calls':case.budget['max_model_calls'],
                            'remaining_generated_tokens':case.budget['max_total_generated_tokens']},
              'encoding_receipt':{'prefix_tokens':prefix,'supervised_tokens':[v['supervised_tokens'] for v in encodings],
                                  'ids_sha256':[identity_digest(v['ids']) for v in encodings]}}
    return record,'matched_executed_contrast'


def audit(args, *, inputs=None):
    import re
    from transformers import AutoTokenizer

    plan,design,data,cases = registered(args) if inputs is None else inputs
    root, out = args.root,args.root/relative(plan['output'])
    target = out/'audit.json'
    controller = out/'controller.json'
    expected = list(schedule(plan['rows']))
    state = read(controller)
    require(not target.exists() and state['phase']=='completed' and state['complete'] and state['backend_stopped']
            and state['frozen_inputs_unchanged'] and state['protocol_sha256']==sha256(args.plan)
            and state['training_only'] is True and not state['optimizer_updates'] and not state['final_test_opened']
            and state['started_epoch']<=state['backend_ready_epoch']<=state['finished_epoch']<=state['deadline_epoch']
            and [r['run_id'] for r in state['runs']]==[name for _,_,name in expected],
            'Expected the complete original collection with an intact controller')
    profile = read(Path(plan['deployment_inputs']['evaluation_profile']))
    student = read(root/relative(read(root/BASELINE)['profile_path']))
    tokenizer = AutoTokenizer.from_pretrained(read(root/RUNTIME)['model'],local_files_only=True,trust_remote_code=False)
    norm = normalization(root)
    journal = RequestJournal(out/'private/backend/requests')
    all_requests, grouped = set(), {}
    for (row,arm,name),result in zip(expected,state['runs'],strict=True):
        case = cases[row['task_id']]
        directory = out/'private/tasks'/name
        identity = {'task_id':row['task_id'],'arm':'control'}
        require(result['sampling_arm']==arm and result['ordinal']==row['ordinal'],'Physical actor belongs to another fork')
        cell,trace,requests = check_arm(root,data,out,directory,case,identity,result,journal,profile,tokenizer,norm)
        require(not all_requests.intersection(requests) and state['backend_ready_epoch']<=trace['started_epoch']
                <=trace['finished_epoch']<=state['finished_epoch'], 'Physical receipts overlap or have an invalid time')
        for index,step in enumerate(trace['steps']):
            received = step['generation'].get('first_action_sampling')
            required = seed_settings(row,design) if index==0 and arm=='sampled' else None
            require(received==required,'Sampling must affect only the first response of the assigned branch')
        all_requests.update(requests)
        grouped.setdefault(row['ordinal'],{})[arm] = (cell,trace,directory)
    require({p.name for p in journal.directory.iterdir()}==all_requests
            and len(all_requests)==sum(value['attempted_calls'] for value in state['runs']),
            'The receipt journal contains an unaccounted generation')
    pairs,records = [],[]
    for row in plan['rows']:
        case = cases[row['task_id']]
        arms = grouped[row['ordinal']]
        cells,traces,dirs = ({arm:value[i] for arm,value in arms.items()} for i in range(3))
        record,contrast = compare_first(case,traces,cells,dirs,tokenizer,profile,student)
        if record is not None:
            text = json.dumps(record,ensure_ascii=False)
            require(not any(re.search(pattern,text) for pattern in SECRET_SHAPES),'Credential-shaped content in private choice data')
            records.append(record)
        pairs.append({'pair':row['ordinal'],'kind':'early_edit','family':case.task['family'],
                      'available_calls':case.budget['max_model_calls'],'arms':cells,'contrast':contrast})
    export = out/'private/preference-pairs.json'
    if records:
        atomic_json(export,records)
    result = {'schema_version':1,'passed':True,'training_only':True,'protocol_sha256':sha256(args.plan),
              'controller_sha256':sha256(controller),'audited_attempts':len(expected),'expected_attempts':len(expected),
              'journal_requests_verified':len(all_requests),'eligible_contrasts':dict(Counter(r['kind'] for r in records)),
              'training_export_sha256':sha256(export) if records else None,'pairs':pairs,'optimizer_updates':0,
              'final_test_opened':False,'weights_public':False}
    atomic_json(target,result)
    return {'audit_sha256':sha256(target),'audited_rollouts':len(expected),'verified_requests':len(all_requests),
            'qualifying_preference_pairs':len(records),'qualifying_families':len({r['provenance']['family'] for r in records}),
            'completed_repairs':{arm:sum(p['arms'][arm]['verified_and_submitted'] for p in pairs) for arm in design['arms']},
            'contrast_counts':dict(Counter(p['contrast'] for p in pairs)),'optimizer_updates':0,'final_test_opened':False}


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','serve','audit'))
    for name in ('root','plan','publication'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--socket',type=Path)
    args = parser.parse_args()
    args.root,args.plan,args.publication = (p.resolve() for p in (args.root,args.plan,args.publication))

    def interrupted(*_):
        raise KeyboardInterrupt('Collection interrupted')

    if args.mode=='serve':
        require(args.socket is not None,'Need a private local socket')
        serve(args)
    elif args.mode=='run':
        for kind in (signal.SIGTERM,signal.SIGINT):
            signal.signal(kind,interrupted)
        collect(args)
    else:
        print(json.dumps(audit(args),allow_nan=False))


if __name__ == '__main__':
    main()
