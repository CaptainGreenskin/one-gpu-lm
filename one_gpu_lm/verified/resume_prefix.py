"""Restore complete 032 observations while retaining an interrupted actor separately."""
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.serving.receipt_client import request_identity
from one_gpu_lm.serving.request_journal import RequestJournal
from .resilient_eval_audit import audit_task
from .tasks import relative


def read(path):
    return json.loads(path.read_text())


def clean_samples(path):
    lines = path.read_bytes().splitlines()
    valid = []
    for line in lines[1:]:
        if b'\0' in line:
            continue
        try:
            row = next(csv.reader([line.decode()]))
            if len(row) != 6:
                continue
            datetime.fromisoformat(row[0]); int(row[1])
            for number in row[2:]:
                float(number)
            valid.append(line)
        except (ValueError, UnicodeError, csv.Error):
            continue
    if not valid:
        raise ValueError('No surviving GPU telemetry')
    return b'\n'.join([lines[0], *valid])+b'\n', len(lines)-len(valid)-1, datetime.fromisoformat(valid[-1].decode().split(',')[0]).timestamp()


def restore_prefix(root, destination, protocol, receipt):
    started = time.time()
    previous = read(root/relative(protocol['interrupted_protocol_path']))
    for key in ('schedule','evaluation_budget','variants','cycle_spec','agent_policy','data_directory'):
        if previous[key] != protocol[key]:
            raise ValueError('Restored evaluation changed its registered inputs')
    prior = root/relative(previous['run_directory'])
    old_controller = prior/'controller.json'
    old_supervisor = root/relative(previous['supervisor_receipt'])
    if (sha256(old_controller) != receipt['interrupted_controller_sha256']
            or sha256(old_supervisor) != receipt['interrupted_supervisor_sha256']):
        raise ValueError('Interrupted process evidence changed')
    state = read(old_controller)
    prefix = state['runs']
    split = receipt['completed_runs']
    old_start = state['started_epoch']
    if (state['complete'] or not old_start < receipt['boot_epoch'] <= started
            or len(prefix) != split or split >= len(protocol['schedule'])
            or [r['run_id'] for r in prefix] != [i['run_id'] for i in protocol['schedule'][:split]]
            or state['current_run'] != protocol['schedule'][split]):
        raise ValueError('Host recovery does not cover the exact completed prefix')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise ValueError('GPU occupied during recovery')

    retained = {entry['index']: set() for entry in state['backend_lifecycles']}
    def directory(index):
        entry = state['backend_lifecycles'][index-1]
        return f"{index:04d}-{entry['role']}"
    for result in prefix:
        run = prior/'private/tasks'/result['run_id']
        trace = read(run/'trajectory.json')
        if not trace['complete'] or read(run/'result.json') != result or sha256(run/'trajectory.json') != result['trajectory_sha256']:
            raise ValueError('Completed pre-reboot trajectory is damaged')
        index = result['backend_index']
        journal = RequestJournal(prior/'private/backends'/directory(index)/'requests')
        retained[index].update(audit_task(run,trace,result,journal))

    split_run = protocol['schedule'][split]['run_id']
    partial = prior/'private/tasks'/split_run/'trajectory.json'
    cut = read(partial)
    if (cut['complete'] or sha256(partial) != receipt['interrupted_trajectory_sha256']
            or split_run != receipt['interrupted_run_id'] or cut['attempted_calls'] != receipt['aborted_attempted_calls']):
        raise ValueError('Aborted task evidence differs')
    index = state['backend_lifecycles'][-1]['index']
    journal = RequestJournal(prior/'private/backends'/directory(index)/'requests')
    observed = [journal.status(request_identity(split_run,call)) for call in range(1,cut['attempted_calls']+1)]
    recorded = [r['result'] for r in observed if r['state']=='completed']
    if (recorded != [s['generation'] for s in cut['steps']]
            or sum(x['generated_tokens'] for x in recorded) != receipt['aborted_known_generated_tokens']
            or sum(x['input_tokens'] for x in recorded) != receipt['aborted_known_input_tokens']
            or sum(r['state']=='in_flight_or_interrupted' for r in observed) != receipt['aborted_unknown_response_count']):
        raise ValueError('Pre-reboot generation costs differ from registered record')

    samples, damaged, last = clean_samples(prior/'telemetry.csv')
    if damaged != receipt['damaged_telemetry_rows'] or last != receipt['last_pre_reboot_gpu_sample_epoch']:
        raise ValueError('Archived GPU sample coverage differs')
    tasks = destination/'private/tasks'; tasks.mkdir(parents=True)
    backends = destination/'private/backends'; backends.mkdir()
    (destination/'telemetry.csv').write_bytes(samples)
    for row in prefix:
        name = row['run_id']
        shutil.copytree(prior/'private/tasks'/name,tasks/name,copy_function=os.link,symlinks=True)
    for entry in state['backend_lifecycles']:
        index = entry['index']; name = directory(index)
        source = prior/'private/backends'/name
        def ignore(path,names):
            if Path(path).name == 'requests':
                return [x for x in names if x not in retained[index]]
            return []
        shutil.copytree(source,backends/name,copy_function=os.link,ignore=ignore)
        os.link(source.with_name(name+'-environment.json'),backends/(name+'-environment.json'))
    last_entry = state['backend_lifecycles'][-1]
    if 'stopped_epoch' in last_entry:
        raise ValueError('Expected a physical backend interrupted by the reboot')
    last_entry.update(stopped_epoch=receipt['boot_epoch'],stop_timestamp_is_upper_bound=True)
    state['resume_preparation_started_epoch'] = started
    state['recovery'] = receipt
    return state
