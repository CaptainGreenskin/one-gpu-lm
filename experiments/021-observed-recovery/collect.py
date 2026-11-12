"""One bounded base rollout per admitted training task; never resume for scores."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.runstate import utc


def collect(root, protocol_path, socket, out, deadline_epoch):
    root, out = Path(root).resolve(), Path(out)
    if out.exists():
        raise ValueError('Do not duplicate training-state collection')
    protocol = json.loads(Path(protocol_path).read_text())
    tasks_path = root / 'private/tasks.json'
    tasks = json.loads(tasks_path.read_text())['tasks']
    training = sorted((t for t in tasks if t['split'] == 'train'), key=lambda t: t['id'])
    if len(training) != 55 or len({t['id'] for t in training}) != 55:
        raise ValueError('Collect every frozen training task exactly once')
    out.mkdir(parents=True, mode=0o700)
    end = min(deadline_epoch, time.time() + protocol['collection']['max_stage_seconds'])
    receipt = {'schema_version': 1, 'complete': False, 'started_utc': utc(),
               'task_spec_sha256': sha256(tasks_path), 'protocol_sha256': sha256(protocol_path),
               'model': 'base', 'tasks': [], 'raw_states_public': False}
    for task in training:
        if time.time() >= end:
            raise TimeoutError('Original collection budget exhausted')
        started = time.monotonic(); process = None; timed_out = False
        command = [str(root / '.venv-agent/bin/python'), str(Path(__file__).with_name('capture.py')),
                   '--root', str(root), '--tasks', str(tasks_path), '--admission', str(root / 'runs/admission/admission.json'),
                   '--protocol', str(protocol_path), '--socket', str(socket), '--out', str(out / task['id']), '--id', task['id']]
        try:
            with (out / (task['id'] + '.log')).open('x') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=stream, start_new_session=True)
                try:
                    code = process.wait(timeout=min(protocol['collection']['task_wall_seconds'], end-time.time()))
                except subprocess.TimeoutExpired:
                    timed_out = True; code = -1
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
        states = sorted((out / task['id'] / 'states').glob('call-*.json'))
        if code == 75 and (len(states) != 8 or json.loads(states[-1].read_text())['completed_calls'] != 8):
            raise ValueError('Interrupted collection lacks its eight completed states')
        row = {'id': task['id'], 'split': 'train', 'returncode': code, 'timed_out': timed_out,
               'seconds': time.monotonic()-started, 'states': {p.name: sha256(p) for p in states},
               'expected_call_boundary': code == 75, 'finished_naturally': code == 0}
        receipt['tasks'].append(row); write_json(out / 'collection.json', receipt)
        print(json.dumps({k: row[k] for k in ['id', 'returncode', 'seconds']}), flush=True)
    receipt.update(complete=True, finished_utc=utc())
    write_json(out / 'collection.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'protocol', 'socket', 'out']: p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--deadline-epoch', type=float, required=True)
    a = p.parse_args(); collect(a.root, a.protocol, a.socket, a.out, a.deadline_epoch)
