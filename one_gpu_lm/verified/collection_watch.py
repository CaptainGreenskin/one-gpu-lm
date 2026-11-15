"""Bound the single registered collector and only its own backend process."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from one_gpu_lm.common.receipts import sha256
from .rollout import atomic
from .tasks import relative


def owns_backend(pid, root, out):
    """Do not signal a reused PID or another user's inference/training process."""
    try:
        proc = Path('/proc')/str(int(pid))
        args = (proc/'cmdline').read_bytes().decode().split('\0')
        return ((proc/'cwd').resolve() == Path(root).resolve()
                and 'one_gpu_lm.serving.local_backend' in args
                and '--out' in args and args[args.index('--out')+1] == str(Path(out)/'private/backend'))
    except (OSError, ValueError, IndexError):
        return False


def run(root, protocol_path):
    root, protocol_path = Path(root).resolve(), Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    out = root/relative(protocol['run_directory'])
    receipt = root/relative(protocol['supervisor_receipt'])
    if out.exists() or receipt.exists():
        raise ValueError('A collection attempt already exists')
    state = {'started_epoch':time.time(), 'protocol_sha256':sha256(protocol_path), 'complete':False}
    with receipt.open('x') as stream:
        stream.write(json.dumps(state)+'\n')
    child = subprocess.Popen(['python3','-u','-m','one_gpu_lm.verified.collection','--root',str(root),
              '--protocol',str(protocol_path)], cwd=root,
              stdout=(root/'private/collection-worker.log').open('wb'), stderr=subprocess.STDOUT,
              start_new_session=True)
    state['worker_pid']=child.pid; atomic(receipt,state)
    timed_out=False
    try:
        child.wait(timeout=protocol['max_pipeline_seconds']+protocol['supervisor_grace_seconds'])
    except subprocess.TimeoutExpired:
        timed_out=True
    # The backend has a separate process group. Always check it even if the
    # controller exits abnormally before its finally block executes.
    controller_path=out/'controller.json'
    controller=json.loads(controller_path.read_text()) if controller_path.exists() else {}
    backend_pid=controller.get('backend_pid')
    state['orphan_backend_cleanup']=False
    if backend_pid and owns_backend(backend_pid,root,out):
        state['orphan_backend_cleanup']=True
        try:
            os.killpg(backend_pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL); child.wait(timeout=10)
    if backend_pid and owns_backend(backend_pid,root,out):
        time.sleep(1)
        if owns_backend(backend_pid,root,out):
            try:
                os.killpg(backend_pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
    state.update(complete=True,timed_out=timed_out,worker_returncode=child.returncode,
                 finished_epoch=time.time(),finished_utc=datetime.now(timezone.utc).isoformat(),
                 worker_result_present=controller_path.exists())
    if timed_out or child.returncode != 0:
        controller.update(stage='supervisor_timeout' if timed_out else 'controller_exit',complete=False,
          finished_epoch=state['finished_epoch'],finished_utc=state['finished_utc'],
          backend_stopped=not backend_pid or not owns_backend(backend_pid,root,out),
          telemetry_may_be_partial=True,final_test_opened=False,development_tasks_scored=0,training_updates=0)
        controller['frozen_files_unchanged']=all((root/relative(n)).is_file() and sha256(root/relative(n))==h
                                                for n,h in protocol['frozen_files'].items())
        out.mkdir(parents=True,exist_ok=True); atomic(controller_path,controller)
    atomic(receipt,state)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True); a=p.parse_args()
    print(json.dumps(run(a.root,a.protocol)))
