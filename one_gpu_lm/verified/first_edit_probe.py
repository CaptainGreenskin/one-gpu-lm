"""Generate one next action per frozen context and checkpoint, with no tool execution."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.serving.receipt_client import generate_once, payload_sha256, request_identity
from one_gpu_lm.serving.request_journal import RequestJournal
from .behavior_diagnosis import SUBMIT, inspection_shaped
from .first_edit_inputs import BASE, VARIANTS, ids_digest, read
from .historic_eval_design import effective_protocol
from .historic_rollout import atomic
from .resilient_backend_control import JournalBackend
from .tasks import relative


def classify(text, case):
    try:
        command = parse_action(text)['command']
    except ValueError:
        return 'invalid'
    if command == case['base_first_edit_command']:
        return 'exact_base_first_edit'
    if command in case['observed_first_edit_commands']:
        return 'other_observed_first_edit'
    if command == case['declared_regression_command']:
        return 'declared_regression'
    if command == SUBMIT:
        return 'submit'
    if inspection_shaped(command):
        return 'inspection_shaped'
    return 'other_parseable'


def committed_inputs(root, protocol_path):
    protocol = read(protocol_path)
    files = protocol['frozen_files']
    if any(sha256(root/relative(name)) != expected for name,expected in files.items()):
        raise ValueError('Frozen probe input or executable changed')
    parent = read(root/relative(protocol['parent_protocol']))
    preparation = read(root/relative(protocol['preparation']))
    private = read(root/relative(protocol['private_states']))['states']
    public = preparation['prepared_states']
    if (preparation['final_test_opened'] or parent['final_test_opened']
            or preparation['source_protocol_sha256'] != files[protocol['parent_protocol']]
            or preparation['private_states_sha256'] != files[protocol['private_states']]
            or protocol['variant_order'] != VARIANTS or protocol['model_generations_per_cell'] != 1):
        raise ValueError('Parent evidence or registered design changed')
    if (len(public) != protocol['states'] or len(private) != len(public)
            or protocol['registered_generations'] != len(public)*len(VARIANTS)
            or len({p['id'] for p in public}) != len(public)):
        raise ValueError('Incomplete or duplicate state inventory')
    cases = []
    for state,case in zip(public,private):
        payload = case['payload']
        if (state['id'] != case['id'] or payload_sha256(payload) != state['payload_sha256']
                or set(payload) != {'operation','messages','agent_policy','max_input_tokens','max_new_tokens'}
                or payload['operation'] != 'generate' or payload['agent_policy'] != parent['agent_policy']
                or payload['max_input_tokens'] != 8192 or not 1 <= payload['max_new_tokens'] <= 512
                or payload['max_new_tokens'] != state['max_generated_tokens']
                or case['base_first_edit_command'] not in case['observed_first_edit_commands']):
            raise ValueError('Recovered development prompt changed')
        cases.append((state,case))
    if protocol['socket_timeout_seconds'] != 180 or protocol['receipt_wait_seconds'] != 180:
        raise ValueError('Changed the per-response timing bounds')
    return protocol,parent,cases


def serving_identity(health,parent,variant):
    return (health.get('ready') is True and health.get('request_journal_version') == 1
            and health['model_id'] == parent['model_id'] and health['revision'] == parent['revision']
            and health['profile_sha256'] == parent['evaluation_profile_sha256']
            and health['adapter'] is (variant != BASE)
            and health['adapter_sha256'] == parent['variants'][variant].get('adapter_sha256',{}))


def run(root,protocol_path):
    protocol,parent,cases = committed_inputs(root,protocol_path)
    publication = read(root/relative(protocol['publication_receipt']))
    if (not publication['anonymous_files_verified'] or publication['ci_conclusion'] != 'success'
            or publication['protocol_sha256'] != sha256(protocol_path)):
        raise ValueError('The registered probe code and protocol require public verification')
    out = root/relative(protocol['run_directory'])
    out.mkdir(parents=True)
    (out/'private').mkdir()
    runtime = read(root/'private/parent-runtime.json')
    profile_path = root/relative(parent['evaluation_profile_path'])
    profile = read(profile_path)
    if profile['chat_template_kwargs'] != {'enable_thinking':False} or profile.get('generation_modes'):
        raise ValueError('Reuse the original 032 greedy non-thinking profile')
    started = time.time()
    deadline = started + protocol['maximum_pipeline_seconds']
    state = {'stage':'verify_model','complete':False,'protocol_sha256':sha256(protocol_path),
             'publication_commit':publication['commit'],'started_epoch':started,
             'started_utc':datetime.fromtimestamp(started,timezone.utc).isoformat(),'deadline_epoch':deadline,
             'runs':[],'backend_lifecycles':[],'model_generations_registered':protocol['registered_generations'],
             'tool_executions':0,'optimizer_updates':0,'final_test_opened':False}

    def save():
        atomic(out/'controller.json',state)

    save()
    telemetry = Telemetry(out/'telemetry.csv',1)
    telemetry.start()
    backend = JournalBackend(root,out,runtime,parent,state['backend_lifecycles'],save)
    try:
        write_json(out/'model-verification.json',verify_model(Path(runtime['model']),
                   root/relative(parent['model_manifest_path']),profile))
        for variant in protocol['variant_order']:
            backend.stop()
            state.update(stage='loading_backend',current_variant=variant,current_state_id=None)
            save()
            backend.protocol = effective_protocol(parent,variant)
            role = 'base' if variant == BASE else 'student'
            socket = backend.ensure(role,deadline)
            backend.entry['variant'] = variant
            backend_directory = backend.journal_path.parent
            if not serving_identity(read(backend_directory/'served-artifact.json'),parent,variant):
                raise ValueError('Unexpected checkpoint served by the backend')
            for registered,case in cases:
                window = protocol['socket_timeout_seconds']+protocol['receipt_wait_seconds']
                if deadline-time.time() < window:
                    raise TimeoutError('Insufficient time for one bounded request')
                state.update(stage='generating',current_state_id=registered['id'])
                save()
                name = variant+'--'+registered['id']
                result,recovery = generate_once(socket,backend.journal_path,name,1,case['payload'],
                    protocol['socket_timeout_seconds'],protocol['receipt_wait_seconds'])
                backend.check_exclusive()
                row = {'state_id':registered['id'],'variant':variant,'request_id':request_identity(name,1),
                       'backend_index':backend.number,'generated_tokens':result['generated_tokens'],
                       'input_tokens':result['input_tokens'],'input_tokens_sha256':result['input_tokens_sha256'],
                       'generation_seconds':result['seconds'],'response_ids_sha256':ids_digest(result['ids']),
                       'recovered_socket_timeout':recovery is not None,
                       'accounting_wait_seconds':recovery['accounting_drain_seconds'] if recovery else 0.0}
                state['runs'].append(row)
                save()
                print(json.dumps({'completed':len(state['runs']),'registered':protocol['registered_generations'],
                                  'variant':variant,'state_id':registered['id']}),flush=True)
        state.update(stage='completed',complete=True)
    except BaseException as error:
        state.update(stage='failed',complete=False,error_type=type(error).__name__)
        (out/'private/probe-error.log').write_text(traceback.format_exc())
    finally:
        backend.stop()
        state['telemetry'] = telemetry.stop()
        end = time.time()
        state.update(finished_epoch=end,finished_utc=datetime.fromtimestamp(end,timezone.utc).isoformat(),
                     backend_stopped=True)
        save()
    return state


def audited_inputs(tokenizer,profile,cases):
    contexts = {}
    for state,case in cases:
        payload = case['payload']
        policy = payload['agent_policy']

        def encode(messages):
            return encode_prefix(tokenizer,messages,profile['chat_template_kwargs'],policy['format'])[0]

        ids,context = prepare_context(payload['messages'],encode,payload['max_input_tokens'],policy['context'])
        if len(ids) != state['input_tokens'] or ids_digest(ids) != state['input_tokens_sha256']:
            raise ValueError('Independent input replay differs from the source evaluation')
        contexts[state['id']] = context
    return contexts


def audit(root,protocol_path):
    from transformers import AutoTokenizer

    protocol,parent,cases = committed_inputs(root,protocol_path)
    directory = root/relative(protocol['run_directory'])
    destination = directory/'audit.json'
    if destination.exists():
        raise ValueError('Preserve the original probe audit')
    state = read(directory/'controller.json')
    if (not state['complete'] or not state['backend_stopped'] or state['protocol_sha256'] != sha256(protocol_path)
            or state['tool_executions'] or state['optimizer_updates'] or state['final_test_opened']):
        raise ValueError('Independent replay requires a completed registered probe')
    profile = read(root/relative(parent['evaluation_profile_path']))
    runtime = read(root/'private/parent-runtime.json')
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False)
    contexts = audited_inputs(tokenizer,profile,cases)
    expected_order = [(variant,state['id']) for variant in VARIANTS for state,_ in cases]
    actual_order = [(row['variant'],row['state_id']) for row in state['runs']]
    if expected_order != actual_order or [b['variant'] for b in state['backend_lifecycles']] != VARIANTS:
        raise ValueError('Probe run order or physical checkpoint loads differ')
    backends = {}
    for entry in state['backend_lifecycles']:
        name = f"{entry['index']:04d}-{entry['role']}"
        backend = directory/'private/backends'/name
        if not serving_identity(read(backend/'served-artifact.json'),parent,entry['variant']):
            raise ValueError('Recorded backend served another checkpoint')
        if read(backend.with_name(name+'-environment.json')) != read(root/relative(parent['environment_path'])):
            raise ValueError('Recorded model interpreter changed')
        backends[entry['index']] = (entry,RequestJournal(backend/'requests'),set())
    case_map = {registered['id']:(registered,case) for registered,case in cases}
    rows = []
    for row in state['runs']:
        registered,case = case_map[row['state_id']]
        entry,journal,seen = backends[row['backend_index']]
        request_id = request_identity(row['variant']+'--'+row['state_id'],1)
        receipt = journal.status(request_id)
        if (row['request_id'] != request_id or request_id in seen or entry['variant'] != row['variant']
                or receipt['state'] != 'completed' or receipt['unknown_generation_cost']
                or receipt['payload_sha256'] != registered['payload_sha256']
                or not entry['ready_epoch'] <= receipt['accepted_epoch'] <= receipt['finished_epoch'] <= entry['stopped_epoch']):
            raise ValueError('Request inventory or durable receipt differs')
        result = receipt['result']
        if (result['input_tokens_sha256'] != row['input_tokens_sha256']
                or row['input_tokens_sha256'] != registered['input_tokens_sha256']
                or result['input_tokens'] != row['input_tokens'] or row['input_tokens'] != registered['input_tokens']
                or result['context'] != contexts[row['state_id']]
                or result['generated_tokens'] != row['generated_tokens']
                or not 1 <= len(result['ids']) == result['generated_tokens'] <= registered['max_generated_tokens']
                or tokenizer.decode(result['ids'],skip_special_tokens=True) != result['text']
                or ids_digest(result['ids']) != row['response_ids_sha256']
                or result['adapter'] is (row['variant'] == BASE)
                or result['seconds'] != row['generation_seconds']):
            raise ValueError('Generation does not match its independently verified receipt and input')
        seen.add(request_id)
        rows.append({**row,'task_id':registered['task_id'],'cohort':registered['cohort'],
                     'history_model':registered['history_model'],'source_call':registered['call'],
                     'action_category':classify(result['text'],case),
                     'same_as_source_output_ids':row['response_ids_sha256']==registered['source_output_ids_sha256']})
    if any({p.name for p in journal.directory.iterdir()} != seen for _,journal,seen in backends.values()):
        raise ValueError('Unexpected physical model request')
    result = {'schema_version':1,'passed':True,'protocol_sha256':sha256(protocol_path),
              'controller_sha256':sha256(directory/'controller.json'),'verified_contexts':len(cases),
              'audited_generations':len(rows),'rows':rows,'audit_model_generations':0,
              'new_tool_executions':0,'optimizer_updates':0,'final_test_opened':False,'weights_public':False}
    write_json(destination,result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['run','audit'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root,protocol_path = args.root.resolve(),args.protocol.resolve()
    result = run(root,protocol_path) if args.phase == 'run' else audit(root,protocol_path)
    passed = result.get('complete') is True if args.phase == 'run' else result['passed']
    print(json.dumps({'phase':args.phase,'passed':passed,'rows':len(result.get('rows',result.get('runs',[])))}))
    raise SystemExit(0 if passed else 1)
