"""Stop only this registered evaluation worker and its socket-owned backend."""
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


def backend_matches(args, cwd, root, socket):
    return (Path(cwd).resolve()==Path(root).resolve()
            and any(module in args for module in ['one_gpu_lm.serving.local_backend','one_gpu_lm.verified.evaluation_backend'])
            and '--socket' in args and args.index('--socket')+1<len(args)
            and args[args.index('--socket')+1]==str(socket))


def owned_backends(root, socket):
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=(proc/'cmdline').read_bytes().decode().split('\0')
            if backend_matches(args,(proc/'cwd').resolve(),root,socket):found.append(int(proc.name))
        except (OSError,UnicodeError,ValueError):continue
    return found


def stop_backends(root, socket, sig):
    stopped=[]
    for pid in owned_backends(root,socket):
        try:os.killpg(pid,sig);stopped.append(pid)
        except ProcessLookupError:pass
    return stopped


def run(root, protocol_path):
    root,protocol_path=Path(root).resolve(),Path(protocol_path).resolve()
    protocol=json.loads(protocol_path.read_text());spec=json.loads((root/relative(protocol['cycle_spec'])).read_text())
    out=root/relative(protocol['run_directory']);receipt=root/relative(protocol['supervisor_receipt'])
    if out.exists() or receipt.exists():raise ValueError('Evaluation attempt already exists')
    state={'started_epoch':time.time(),'protocol_sha256':sha256(protocol_path),'complete':False}
    with receipt.open('x') as stream:stream.write(json.dumps(state)+'\n')
    child=subprocess.Popen(['python3','-u','-m','one_gpu_lm.verified.evaluation','--root',str(root),'--protocol',str(protocol_path)],
              cwd=root,stdout=(root/'private/factorial-worker.log').open('wb'),stderr=subprocess.STDOUT,start_new_session=True)
    state['worker_pid']=child.pid;atomic(receipt,state);timed_out=False
    try:child.wait(timeout=spec['development']['maximum_evaluation_pipeline_seconds']+30)
    except subprocess.TimeoutExpired:timed_out=True
    # Discover by this run's exact private socket as well as cwd, so even a
    # worker killed between spawn and receipt persistence cannot orphan a GPU.
    socket=out/'backend.sock';state['orphan_backends_terminated']=stop_backends(root,socket,signal.SIGTERM)
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
    time.sleep(1)
    state['orphan_backends_killed']=stop_backends(root,socket,signal.SIGKILL)
    path=out/'controller.json';controller=json.loads(path.read_text()) if path.exists() else {}
    state.update(complete=True,timed_out=timed_out,worker_returncode=child.returncode,
                 finished_epoch=time.time(),finished_utc=datetime.now(timezone.utc).isoformat())
    if timed_out or child.returncode!=0 or state['orphan_backends_terminated'] or state['orphan_backends_killed']:
        controller.update(complete=False,stage='supervisor_timeout' if timed_out else 'unexpected_worker_exit',
          finished_epoch=state['finished_epoch'],finished_utc=state['finished_utc'],backend_stopped=not owned_backends(root,socket),
          telemetry_may_be_partial=True,final_test_opened=False,weights_public=False)
        out.mkdir(parents=True,exist_ok=True);atomic(path,controller)
    state['evaluation_complete']=bool(controller.get('complete')) and not timed_out and child.returncode==0
    atomic(receipt,state)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.root,a.protocol)))
