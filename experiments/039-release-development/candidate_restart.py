"""Recover the exact committed development prefix after a Linux reboot."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json
from one_gpu_lm.serving.receipt_client import payload_sha256, request_identity
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.resilient_eval_audit import audit_task
from one_gpu_lm.verified.tasks import relative


def read(path):
    return json.loads(path.read_bytes())


def host_boot():
    identity = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    started = next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
    return identity, int(started)


def attempt_directory(out, attempt):
    if type(attempt) is not int or attempt < 1:
        raise ValueError('Invalid evaluation attempt number')
    return out if attempt == 1 else out/'private/attempts'/f'{attempt:03d}'


def actor_directory(out, attempt, run_id):
    return attempt_directory(out, attempt)/'private/tasks'/relative(run_id)


def backend_directory(out, entry):
    return attempt_directory(out, entry.get('attempt', 1))/'private/backends'/f"{entry['index']:04d}-{entry['role']}"


def completed_actor(out, result, entry):
    actor = actor_directory(out, result.get('attempt', 1), result['run_id'])
    path = actor/'trajectory.json'
    trace = read(path)
    served = read(backend_directory(out, entry)/'served-artifact.json')
    if (read(actor/'result.json') != result or sha256(path) != result['trajectory_sha256']
            or trace['complete'] is not True or trace['backend'] != served
            or result.get('attempt', 1) != entry.get('attempt', 1)
            or not entry['ready_epoch'] <= trace['started_epoch'] <= trace['finished_epoch'] <= entry['stopped_epoch']):
        raise ValueError('A completed pre-reboot task changed')
    directory = backend_directory(out, entry)/'requests'
    if not directory.is_dir():
        raise ValueError('Completed pre-reboot task lost its request journal')
    requests = audit_task(actor, trace, result, RequestJournal(directory))
    if len(requests) != result['attempted_calls']:
        raise ValueError('A completed pre-reboot task is missing its journal receipts')
    return requests


def interrupted_actor(out, prior, item, backends, boot_epoch, max_calls):
    if item is None:
        return None, set()
    actor = actor_directory(out, prior['attempt'], item['run_id'])
    trace_path = actor/'trajectory.json'
    result_path = actor/'result.json'
    trace = read(trace_path) if trace_path.is_file() else {}
    index = prior.get('current_backend_index')
    attempted = trace.get('attempted_calls', 0)
    if type(attempted) is not int or not 0 <= attempted <= max_calls:
        raise ValueError('Interrupted trajectory exceeds its original call budget')
    if not trace and not actor.exists():
        return None, set()
    requests, complete = set(), {}
    unknown = 0
    if trace:
        entry = backends[index]
        directory = backend_directory(out, entry)
        if (entry.get('attempt', 1) != prior['attempt'] or trace['backend'] != read(directory/'served-artifact.json')
                or not entry['ready_epoch'] <= trace['started_epoch'] <= boot_epoch):
            raise ValueError('Interrupted actor used a different backend')
        if not (directory/'requests').is_dir():
            raise ValueError('Interrupted actor lost its request journal')
        journal = RequestJournal(directory/'requests')
        for call in range(1, attempted + 1):
            request_id = request_identity(item['run_id'], call)
            receipt = journal.status(request_id)
            if receipt['state'] == 'not_received':
                continue
            requests.add(request_id)
            if 'accepted_epoch' in receipt:
                payload = read(actor/f'request-{call:03d}.json')
                if (receipt['request_id'] != request_id or receipt['payload_sha256'] != payload_sha256(payload)
                        or not trace['started_epoch'] <= receipt['accepted_epoch'] <= receipt.get('finished_epoch', boot_epoch) <= boot_epoch):
                    raise ValueError('Interrupted journal did not belong to the preserved request')
            if receipt['state'] == 'completed':
                if receipt['unknown_generation_cost'] or 'accepted_epoch' not in receipt:
                    raise ValueError('Completed interrupted response lacks a cost receipt')
                complete[call] = receipt['result']
            else:
                unknown += 1
        if any(complete.get(step['call']) != step['generation'] for step in trace['steps']):
            raise ValueError('Interrupted actor steps differ from their backend receipts')
    return {'backend_index': index, 'attempted_calls': attempted, 'journaled_requests': len(requests),
            'finished_responses': len(complete), 'unknown_response_costs': unknown,
            'known_input_tokens': sum(value['input_tokens'] for value in complete.values()),
            'known_generated_tokens': sum(value['generated_tokens'] for value in complete.values()),
            'trajectory_sha256': sha256(trace_path) if trace_path.is_file() else None,
            'result_sha256': sha256(result_path) if result_path.is_file() else None}, requests


def backend_index(state):
    entries = {entry['index']: entry for entry in state['backend_lifecycles']}
    if len(entries) != len(state['backend_lifecycles']):
        raise ValueError('Duplicate loaded model identity')
    return entries


def reconcile_requests(out, state, scored, interrupted):
    journals = {index: backend_directory(out, entry)/'requests' for index, entry in backend_index(state).items()
                if 'ready_epoch' in entry}
    physical = {path for number in range(1, state.get('attempt', 1)+1)
                for path in (attempt_directory(out, number)/'private/backends').glob('*/requests')
                if any(path.iterdir())}
    if (not physical <= set(journals.values()) or not set(scored) <= journals.keys()
            or not set(interrupted) <= journals.keys()):
        raise ValueError('Unaccounted backend journal entries')
    for index, path in journals.items():
        used = scored.get(index, set())
        cut = interrupted.get(index, set())
        if not path.is_dir() or used & cut or used | cut != {entry.name for entry in path.iterdir()}:
            raise ValueError('Unaccounted backend journal entries')


def prepare_restart(out, prior, plan, protocol_hash, order, boot):
    """Return a new controller without replacing any prior task or journal file."""
    now = time.time()
    boot_id, boot_epoch = boot
    attempt = prior['attempt']
    count = len(prior['runs'])
    possible_current = order[max(0,count-1):count+1]
    if (prior['boot_id'] == boot_id or prior['complete'] or prior['stage'] not in ('loading', 'evaluating')
            or prior['protocol_sha256'] != protocol_hash or prior['schedule_sha256'] != plan['schedule_sha256']
            or prior['final_test_opened'] or prior['weights_public'] or prior['optimizer_updates']
            or prior['deadline_epoch'] != prior['started_epoch'] + plan['pipeline_limit_seconds']
            or not prior['attempt_started_epoch'] <= boot_epoch <= now < prior['deadline_epoch']
            or [row['run_id'] for row in prior['runs']] != [item['run_id'] for item in order[:count]]
            or (prior.get('current_run') is not None and prior['current_run'] not in possible_current)
            or attempt != len(prior['restarts']) + 1):
        raise ValueError('Require a rebooted incomplete prefix under the original deadline')
    state = deepcopy(prior)
    backends = backend_index(state)
    for entry in backends.values():
        if 'stopped_epoch' not in entry:
            if entry.get('attempt', 1) != attempt:
                raise ValueError('A previous backend lacks a stop boundary')
            entry.update(stopped_epoch=boot_epoch, stop_timestamp_is_upper_bound=True)
    scored = defaultdict(set)
    for row in state['runs']:
        index = row['backend_index']
        scored[index].update(completed_actor(out, row, backends[index]))

    pending = order[count] if count < len(order) and prior.get('current_run') == order[count] else None
    recovered = False
    if pending is not None:
        result_path = actor_directory(out, attempt, pending['run_id'])/'result.json'
        result = read(result_path) if result_path.is_file() else {}
        if (result.get('run_id') == pending['run_id'] and result.get('attempt') == attempt
                and result.get('backend_index') == prior.get('current_backend_index')
                and result['contrast_arm'] == pending['arm']
                and all(result[key] == pending[key] for key in ('task_number', 'repetition', 'block'))):
            index = result['backend_index']
            scored[index].update(completed_actor(out, result, backends[index]))
            state['runs'].append(result)
            recovered = True
    interrupted, _ = interrupted_actor(out, prior, None if recovered else pending, backends, boot_epoch,
                                      plan['evaluation_budget']['max_model_calls'])
    directory = attempt_directory(out, attempt)
    snapshot = directory/'closed-controller.json'
    if snapshot.is_file():
        if read(snapshot) != prior:
            raise ValueError('Preserve the previously recorded interrupted controller')
    else:
        atomic_json(snapshot, prior)
    telemetry = directory/'telemetry.csv'
    state['restarts'].append({'prior_attempt': attempt, 'boot_epoch': boot_epoch, 'resumed_epoch': now,
                              'completed_prefix': count, 'recovered_committed_result': recovered,
                              'pending_run': None if recovered else pending, 'interrupted': interrupted,
                              'prior_controller_sha256': sha256(snapshot),
                              'prior_telemetry_sha256': sha256(telemetry) if telemetry.is_file() else None})
    state.update(attempt=attempt+1, boot_id=boot_id, attempt_started_epoch=now,
                 current_run=None, current_backend_index=None, stage='loading')
    extra, _ = replay_interruptions(out, state, order, plan['evaluation_budget']['max_model_calls'])
    reconcile_requests(out, state, scored, extra)
    return state


def interruption_summary(state):
    events = state.get('restarts', [])
    costs = [event['interrupted'] for event in events if event['interrupted'] is not None]
    metrics = ('attempted_calls','journaled_requests','finished_responses','unknown_response_costs',
               'known_input_tokens','known_generated_tokens')
    return {'reboots': len(events), 'retained_prefixes': [event['completed_prefix'] +
            int(event['recovered_committed_result']) for event in events],
            **{key: sum(row[key] for row in costs) for key in metrics}}


def replay_interruptions(out, state, order, max_calls):
    """Audit the interrupted physical requests separately from the scored replay."""
    extra = defaultdict(set)
    backends = backend_index(state)
    if state.get('attempt',1) != len(state.get('restarts',[])) + 1:
        raise ValueError('Incomplete history of GPU evaluation attempts')
    for number, event in enumerate(state.get('restarts', []), 1):
        directory = attempt_directory(out, number)
        path = directory/'closed-controller.json'
        prior = read(path)
        telemetry = directory/'telemetry.csv'
        count = event['completed_prefix']
        pending = order[count] if count < len(order) and prior.get('current_run') == order[count] else None
        if event['recovered_committed_result']:
            if pending is None or state['runs'][count]['run_id'] != pending['run_id'] or state['runs'][count]['attempt'] != number:
                raise ValueError('Recovered terminal result differs from the scheduled task')
            pending = None
        if (event['prior_attempt'] != prior['attempt'] or prior['attempt'] != number or sha256(path) != event['prior_controller_sha256']
                or prior['protocol_sha256'] != state['protocol_sha256'] or prior['deadline_epoch'] != state['deadline_epoch']
                or not prior['attempt_started_epoch'] <= event['boot_epoch'] <= event['resumed_epoch'] < state['deadline_epoch']
                or prior['runs'] != state['runs'][:count] or event['pending_run'] != pending
                or (sha256(telemetry) if telemetry.is_file() else None) != event['prior_telemetry_sha256']):
            raise ValueError('Previous attempt receipts differ from the recorded reboot')
        expected, requests = interrupted_actor(out, prior, pending, backends, event['boot_epoch'], max_calls)
        if expected != event['interrupted']:
            raise ValueError('Interrupted generation accounting changed')
        if requests:
            index = expected['backend_index']
            if extra[index] & requests:
                raise ValueError('An interrupted generation is recorded twice')
            extra[index].update(requests)
    return extra, interruption_summary(state)
