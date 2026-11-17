"""Fresh base-teacher collection with fully recorded correction-policy feedback."""
import base64
import copy
import json
from pathlib import Path
import shutil
import time

from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import format_feedback
from one_gpu_lm.common.receipts import write_json
from one_gpu_lm.serving.client import request
from .correction_policy import initial_messages, CorrectionGuard, feedback_notice
from .decision_policy import guard_arm, notice as continuation_notice
from .evaluation_rollout import atomic, normalization
from .recovery_state import remaining_budget, canonical
from .recovery_rollout import identity_ok, finish
from .submission import SubmissionGuard, SUBMIT
from .tasks import prepare, grade, files, digest


def initialize(data, task, source, protocol, out, arm):
    out = Path(out)
    if task['split'] != 'training' or out.exists():
        raise ValueError('New training-only attempts required')
    out.mkdir(parents=True)
    before = prepare(Path(data)/'snapshots'/task['repository'].replace('/', '--'), out/'workspace', source, task)
    state = {'schema_version': 1, 'id': task['id'], 'repository': task['repository'], 'family': task['family'],
             'split': 'training', 'branch': arm, 'producer_kind': 'model',
             'protocol_sha256': digest(json.dumps(protocol, sort_keys=True).encode()),
             'started_epoch': time.time(), 'complete': False, 'boundary': 'before_generation',
             'messages': initial_messages(task, protocol, arm), 'steps': [], 'backend_identities': {},
             'attempted_calls': 0, 'generated_tokens': 0, 'actor_seconds': 0., 'submitted': False,
             'terminal_reason': None, 'verified_source': None,
             'original_source_sha256': {n: before[n] for n in task['editable_files']}}
    write_json(out/'before.json', before)
    atomic(out/'trajectory.json', state)
    return state


def advance(root, task, protocol, out, state, backend, runner, arm):
    """Continue only from a known feedback boundary, with cumulative path costs."""
    out = Path(out); budget = protocol['evaluation_budget']; policy = protocol['agent_policy']
    if (state['complete'] or state.get('unknown_inflight_generation_cost')
            or state['boundary'] not in {'before_generation', 'completed_step'}):
        raise ValueError('Do not resume unknown or terminal actor state')
    if state['attempted_calls'] != 0 or state['branch'] != arm:
        raise ValueError('Fresh complete attempts only; never resume or retry')
    role = 'base'
    if state['terminal_reason']:
        return state
    health = request(backend, {'operation': 'health'})
    if not identity_ok(health, protocol, role):
        raise ValueError('Unexpected continuation artifact')
    state['backend_identities'][role] = health
    norm = normalization(root)
    workspace = out/'workspace'; key = str(workspace.resolve())
    guard = CorrectionGuard(runner, {key: {k: task[k] for k in ['editable_files', 'regression_command']}}, guard_arm(protocol, arm))
    original_inventory = json.loads((out/'before.json').read_text())
    if state.get('verified_source') is not None:
        if state['verified_source'] != guard.source_state(workspace, task):
            raise ValueError('Saved verification is stale')
        guard.verified[key] = copy.deepcopy(state['verified_source'])
    consumed = state['actor_seconds']; started = time.monotonic()
    def save():
        state['actor_seconds'] = consumed + time.monotonic() - started
        atomic(out/'trajectory.json', state)
    for call in range(state['attempted_calls']+1, budget['max_model_calls']+1):
        state['actor_seconds'] = consumed + time.monotonic() - started
        remaining = remaining_budget(budget, state)
        if remaining['actor_seconds'] <= 0 or remaining['generated_tokens'] <= 0:
            state['terminal_reason'] = 'time_limit' if remaining['actor_seconds'] <= 0 else 'token_limit'
            break
        payload = {'operation': 'generate', 'messages': copy.deepcopy(state['messages']), 'agent_policy': policy,
                   'max_input_tokens': budget['max_input_tokens'], 'max_new_tokens': remaining['next_generation_cap']}
        request_path = out/f'request-{call:03d}.json'
        if request_path.exists():
            raise ValueError('No duplicate generation request')
        write_json(request_path, payload)
        state.update(boundary='in_flight', attempted_calls=call); save()
        try:
            generated = request(backend, payload, timeout=max(1, min(180, remaining['actor_seconds'])))
        except Exception as error:
            state.update(terminal_reason='backend_error', backend_error_type=type(error).__name__,
                         unknown_inflight_generation_cost=True)
            save()
            raise RuntimeError('Stop pilot after unaccounted generation; no retry') from None
        state['generated_tokens'] += generated['generated_tokens']
        evidence_before, protected_before = save_source(out, workspace, task, original_inventory)
        step = {'source_evidence_before': evidence_before, 'protected_before': protected_before, 'call': call, 'model_role': role, 'request_sha256': digest(request_path.read_bytes()),
                'generation': generated, 'source_before': guard.source_state(workspace, task), 'action_format_valid': False}
        assistant = {'role': 'assistant', 'content': generated['text'], 'extra': {}}
        state['messages'].append(assistant)
        try:
            action = parse_action(generated['text'])
        except ValueError:
            state['messages'].append({'role': 'user', 'content': format_feedback(policy)})
            state['steps'].append(step); state['boundary'] = 'completed_step'; save(); continue
        step.update(action_format_valid=True, command=action['command']); assistant['extra']['actions'] = [action]
        seconds = budget['max_task_seconds'] - (consumed + time.monotonic() - started)
        if seconds <= 0:
            state['steps'].append(step); state['terminal_reason'] = 'time_limit_before_tool'; break
        guard.timeout = max(1, min(budget['max_tool_seconds'], int(seconds)))
        output = guard.execute(workspace, action['command'])
        write_json(out/f'tool-{call:03d}.json', output)
        raw = output['output']; cap = budget['max_tool_output_chars']
        delivered, receipt = norm.transform(raw if len(raw) <= cap else raw[:cap]+norm.MARKER, 'extended', cap)
        step.update(observation={k: v for k, v in output.items() if k != 'output'}, normalization=receipt)
        try:
            after = files(workspace); step['source_after'] = {n: after.get(n) for n in task['editable_files']}
            step['source_evidence_after'], step['protected_after'] = save_source(out, workspace, task, original_inventory)
        except ValueError:
            state['steps'].append(step); state['terminal_reason'] = 'unsafe_workspace'; break
        notice = continuation_notice(protocol, arm, task, state['steps'], step)
        step['policy_notice'] = notice
        state['messages'].append({'role': 'user', 'content': f"Exit code: {output['returncode']}\n{delivered}{notice}",
                                  'extra': {'tool': step['observation']}})
        state['verified_source'] = copy.deepcopy(guard.verified.get(key))
        state['steps'].append(step); state['boundary'] = 'completed_step'; save()
        if action['command'] == SUBMIT and output['returncode'] == 0:
            state.update(submitted=True, terminal_reason='submitted'); break
    if state['terminal_reason'] is None and state['attempted_calls'] >= budget['max_model_calls']:
        state['terminal_reason'] = 'call_limit'
    save()
    return state



def save_source(out, workspace, task, original_inventory):
    inventory = files(workspace)
    sources = {n: inventory.get(n) for n in task['editable_files']}
    payload = {'source_sha256': sources, 'workspace_inventory': inventory, 'contents_base64': {
        n: base64.b64encode((Path(workspace)/n).read_bytes()).decode() if sources[n] is not None else None
        for n in task['editable_files']}}
    data = canonical(payload); fingerprint = digest(data)
    directory = Path(out)/'source-states'; directory.mkdir(exist_ok=True)
    path = directory/(fingerprint+'.json')
    if path.exists() and path.read_bytes() != data: raise ValueError('Source evidence collision')
    if not path.exists(): path.write_bytes(data)
    protected = all(inventory.get(n) == h for n,h in original_inventory.items() if n not in task['editable_files'])
    return fingerprint, protected
