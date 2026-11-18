"""Exercise deadline recovery and task continuation on GPU using synthetic tasks."""
import argparse
import json
from pathlib import Path
import time

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.receipt_client import payload_sha256, request_identity
from one_gpu_lm.serving.request_journal import RequestJournal
from .historic_eval_design import effective_protocol
from .resilient_backend_control import JournalBackend
from .resilient_rollout import run as rollout
from .sandbox import Sandbox
from .tasks import files


def run(root, protocol_path, out):
    read = lambda path: json.loads(Path(path).read_text())
    protocol = read(protocol_path)
    runtime = read(root/'private/parent-runtime.json')
    effective = effective_protocol(protocol, 'multifile032')
    runtime_map = read(root/protocol['runtime_map'])
    sandbox = Sandbox(read(runtime_map['dateutil/dateutil']), 30)
    out.mkdir(parents=True)
    data = out/'fixture'
    repository = 'fixture/arith'
    revision = 'b'*40
    source_id = 'fixture--arith--'+revision
    snapshot = data/'snapshots'/source_id
    snapshot.mkdir(parents=True)
    (snapshot/'value.py').write_text('VALUE = 2\n')
    (snapshot/'LICENSE').write_text('Synthetic fixture made for this repository\n')
    source = {'source_id': source_id, 'repository': repository, 'family': repository,
              'revision': revision, 'upstream_parent': 'a'*40, 'split': 'development',
              'approved_for': {'development': True}, 'notice_review_complete': True,
              'lineage_review_complete': True, 'notices': ['LICENSE'], 'files': files(snapshot)}
    task = {**{k: source[k] for k in ('source_id','repository','family','revision','upstream_parent','split')},
            'origin': 'upstream_history_parent_plus_test_patch',
            'description': 'First run the declared regression command as the entire first bash tool call. Then repair the integer constant in value.py.',
            'editable_files': ['value.py'],
            'mutations': [{'path':'value.py','before':'VALUE = 2','after':'VALUE = 1'}],
            'visible_checks': {'test_value.py': 'from value import VALUE\nassert VALUE == 2\n'},
            'hidden_checks': {}, 'regression_command': 'python3 test_value.py',
            'grading_command': 'python3 test_value.py'}
    started = time.time()
    lifecycles = []
    backend = JournalBackend(root, out, runtime, effective, lifecycles)
    try:
        socket = backend.ensure('student', started+600)
        pid = backend.process.pid
        baseline_hash = effective['adapter_sha256']
        traces = []
        for name, seconds in [('synthetic-timeout', .25), ('synthetic-next', 60)]:
            directory = out/'private/tasks'/name
            run_protocol = dict(effective, evaluation_budget=dict(effective['evaluation_budget'],
                                max_task_seconds=seconds, max_model_calls=1,
                                max_generated_tokens_per_call=96, max_total_generated_tokens=96))
            result = rollout(root, data, dict(task, id=name), source, socket, sandbox, run_protocol,
                             directory, 'student_current', journal_path=backend.journal_path,
                             recovery_seconds=180)
            backend.check_exclusive()
            trace = read(directory/'trajectory.json')
            journal = RequestJournal(backend.journal_path)
            receipt = journal.status(request_identity(name,1))
            step = trace['steps'][0]
            checks = {
                'same_gpu_backend': backend.process.pid == pid and len(lifecycles) == 1,
                'same_032_adapter': trace['backend']['adapter_sha256'] == baseline_hash,
                'positive_gpu_generation': step['generation']['generated_tokens'] > 0
                    and step['generation']['peak_allocated_bytes'] > 0,
                'exact_durable_response': receipt['state'] == 'completed'
                    and not receipt['unknown_generation_cost'] and receipt['result'] == step['generation']
                    and receipt['payload_sha256'] == payload_sha256(read(directory/'request-001.json')),
                'exact_one_call': result['complete'] and trace['attempted_calls'] == len(trace['steps']) == 1,
                'no_duplicate_generation': len(list(journal.directory.iterdir())) == len(traces)+1,
            }
            if not traces:
                checks.update(timeout_recovered=trace.get('timeout_call') == 1,
                              no_late_tool='observation' not in step and not list(directory.glob('tool-*.json')),
                              no_late_submission=not result['submitted'] and not result['complete_success_within_budget'],
                              original_deadline_exceeded=trace['actor_seconds'] >= seconds)
            else:
                checks.update(next_completed_without_timeout=trace.get('timeout_call') is None,
                              next_tool_executed='observation' in step)
            traces.append({'id':name, 'actor_limit_seconds':seconds, 'actor_seconds':trace['actor_seconds'],
                           'generated_tokens':trace['generated_tokens'], 'input_tokens':step['generation']['input_tokens'],
                           'model_seconds':step['generation']['seconds'],
                           'accounting_wait_seconds':trace.get('accounting_drain_seconds',0),
                           'trajectory_sha256':sha256(directory/'trajectory.json'), 'checks':checks})
            if not all(checks.values()):
                break
        result = {'passed':len(traces)==2 and all(all(t['checks'].values()) for t in traces),
                  'source':'synthetic_fixture_no_registered_development_tasks',
                  'started_epoch':started, 'tasks':traces, 'final_test_opened':False}
    finally:
        backend.stop()
    result.update(finished_epoch=time.time(), backend_stopped=True,
                  backend_loads=len(lifecycles), adapter_sha256=baseline_hash)
    write_json(out/'result.json',result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','protocol','out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    result = run(a.root.resolve(),a.protocol.resolve(),a.out.resolve())
    print(json.dumps(result))
