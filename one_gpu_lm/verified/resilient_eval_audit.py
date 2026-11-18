"""Check every journal entry against the existing tokenizer and grader replay."""
import argparse
import json
from pathlib import Path

from one_gpu_lm.common.receipts import write_json
from one_gpu_lm.serving.receipt_client import payload_sha256, request_identity
from one_gpu_lm.serving.request_journal import RequestJournal
from .historic_eval_audit import run as replay_evaluation


def audit_task(directory, trace, result, journal):
    request_ids = set()
    recovered = []
    for step in trace['steps']:
        call = step['call']
        request_id = request_identity(result['run_id'], call)
        payload = json.loads((directory/f'request-{call:03d}.json').read_text())
        receipt = journal.status(request_id)
        if (receipt.get('state') != 'completed' or receipt['unknown_generation_cost']
                or receipt['request_id'] != request_id or request_id in request_ids
                or receipt['payload_sha256'] != payload_sha256(payload)
                or receipt['result'] != step['generation']
                or not trace['started_epoch'] <= receipt['accepted_epoch'] <= receipt['finished_epoch'] <= trace['finished_epoch']):
            raise ValueError('Journal does not identify one complete recorded response')
        request_ids.add(request_id)
        late = step.get('transport_timeout_receipt')
        if late is not None:
            final_source = {name:trace['final_source_sha256'].get(name) for name in step['source_before']}
            if (late['request_id'] != request_id or late['accepted_epoch'] != receipt['accepted_epoch']
                    or late['finished_epoch'] != receipt['finished_epoch']
                    or 'observation' in step or (directory/f'tool-{call:03d}.json').exists()
                    or step['source_before'] != final_source):
                raise ValueError('Late response altered actor state or differs from durable receipt')
            recovered.append((call, late['accounting_drain_seconds']))
    if recovered:
        if (recovered != [(trace['attempted_calls'], result['accounting_drain_seconds'])]
                or trace.get('timeout_call') != result['timeout_call']
                or result['timeout_call'] != trace['attempted_calls']
                or result['accounting_drain_seconds'] != trace.get('accounting_drain_seconds')
                or not 0 <= result['accounting_drain_seconds'] <= trace['actor_seconds']
                or trace['terminal_reason'] not in {'time_limit', 'time_limit_before_tool'}
                or trace['submitted'] or result['complete_success_within_budget']):
            raise ValueError('Recovered timeout did not end the task without submission')
    elif result['timeout_call'] is not None or result['accounting_drain_seconds'] != 0:
        raise ValueError('Timeout summary lacks a recovered response')
    return request_ids


def run(root, protocol_path, out):
    if out.exists():
        raise ValueError('Preserve the prior audit')
    protocol = json.loads(protocol_path.read_text())
    directory = root/protocol['run_directory']
    base = replay_evaluation(root, protocol_path, directory/'original-audit-checks.json')
    controller = json.loads((directory/'controller.json').read_text())
    raw_runs = {r['run_id']: r for r in controller['runs']}
    backends = {r['index']:r for r in controller['backend_lifecycles']}
    journals = {index:RequestJournal(directory/'private/backends'/f"{index:04d}-{entry['role']}"/'requests')
                for index,entry in backends.items() if 'ready_epoch' in entry}
    seen = {index:set() for index in journals}
    for row in base['rows']:
        if not row['complete']:
            continue
        raw = raw_runs[row['run_id']]
        index = raw['backend_index']
        task = directory/'private/tasks'/row['run_id']
        trace = json.loads((task/'trajectory.json').read_text())
        try:
            entry = backends[index]
            if (entry['variant'] != row['variant'] or trace['backend'].get('request_journal_version') != 1
                    or not entry['ready_epoch'] <= trace['started_epoch'] <= trace['finished_epoch'] <= entry['stopped_epoch']):
                raise ValueError('Misbound physical backend')
            ids = audit_task(task, trace, raw, journals[index])
            if seen[index] & ids:
                raise ValueError('A physical generation request appeared in more than one task')
            seen[index].update(ids)
            row.update(journal_audit_passed=True, timeout_call=raw['timeout_call'],
                       accounting_drain_seconds=raw['accounting_drain_seconds'])
        except (KeyError, ValueError, OSError, TypeError):
            row.update(journal_audit_passed=False, audit_passed=False)
    inventory = sum(len(s) for s in seen.values())
    base['checks'].update(all_task_journals_verified=all(r.get('journal_audit_passed') for r in base['rows']),
        journal_inventory_exact=all({p.name for p in journals[i].directory.iterdir()} == ids for i,ids in seen.items()),
        physical_request_count_matches_journal=inventory == sum(x['attempted_calls'] for x in base['physical_attempts']))
    base['passed'] = all(base['checks'].values())
    base['journal'] = {'verified_generation_requests':inventory,
        'recovered_timeout_tasks':sum(r.get('timeout_call') is not None for r in base['rows']),
        'accounting_drain_seconds':sum(r.get('accounting_drain_seconds',0) for r in base['rows'])}
    write_json(out,base)
    return base


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','protocol','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    result = run(a.root.resolve(),a.protocol.resolve(),a.out.resolve())
    print(json.dumps({k:result[k] for k in ('passed','checks','journal')}))
