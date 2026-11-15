"""Bound the registered capacity worker's lifetime without competing GPU jobs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from one_gpu_lm.common.receipts import sha256, write_json
from .capacity import validate_protocol
from .tasks import relative


def run(root, protocol_path):
    root, protocol_path = Path(root).resolve(), Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text()); validate_protocol(protocol)
    out = root / relative(protocol['run_directory'])
    receipt = root / 'private/capacity-supervisor.json'
    if out.exists() or receipt.exists():
        raise ValueError('An attempt already exists; never duplicate capacity compute')
    runtime = json.loads((root/'private/parent-runtime.json').read_text())
    state = {'started_epoch': time.time(), 'protocol_sha256': sha256(protocol_path), 'complete': False}
    # Exclusive receipt creation prevents two supervisors racing before the worker
    # creates its result directory.
    with receipt.open('x') as stream:
        stream.write(json.dumps(state)+'\n')
    log = (root/'private/capacity-worker.log').open('wb')
    child = subprocess.Popen([runtime['model_python'], '-u', '-m', 'one_gpu_lm.verified.capacity',
                              '--root', str(root), '--protocol', str(protocol_path)],
                             cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state['worker_pid'] = child.pid; write_json(receipt, state)
    timed_out = False
    try:
        child.wait(timeout=protocol['maximum_seconds'])
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
    state.update(complete=True, timed_out=timed_out, worker_returncode=child.returncode,
                 finished_epoch=time.time(), finished_utc=datetime.now(timezone.utc).isoformat())
    result_path = out/'result.json'
    if timed_out:
        value = json.loads(result_path.read_text()) if result_path.exists() else {}
        value.update(stage='wall_clock_timeout', complete=False, capacity_gate_passed=False,
                     finished_epoch=state['finished_epoch'], finished_utc=state['finished_utc'],
                     telemetry_may_be_partial=True)
        out.mkdir(parents=True, exist_ok=True)
        write_json(result_path, value)
    state['worker_result_present'] = result_path.exists()
    write_json(receipt, state)
    return state


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True); a=p.parse_args()
    value=run(a.root, a.protocol)
    print(json.dumps({k:value[k] for k in ['complete','timed_out','worker_returncode','worker_result_present']}))
