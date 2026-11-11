"""Run a preregistered paired tool-history diagnostic on one unchanged local model."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, gpu_lock_path, single_writer, utc
from one_gpu_lm.common.source import campaign_sources
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.serving.client import request
from one_gpu_lm.evaluation.tasks import file_manifest


def terminate(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def schedule(ids, variants):
    return [(task, arm) for i, task in enumerate(ids)
            for arm in variants[i % len(variants):] + variants[:i % len(variants)]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    os.chdir(root)
    exp = root / 'experiments/017-edit-progress'
    protocol_path = exp / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if protocol['status'] != 'frozen_before_model_rollouts':
        raise ValueError('Freeze controls and policies before evaluation')
    if out.exists():
        raise ValueError('Preserve existing study; inspect boundary before explicit recovery')
    for path, digest in protocol['frozen_files'].items():
        if sha256(root / path) != digest:
            raise ValueError('Changed frozen input: ' + path)
    snapshots = json.loads((root/'source-snapshots.json').read_text())
    for slug, expected in snapshots.items():
        if file_manifest((root/'sources'/slug).resolve()) != expected:
            raise ValueError('Pinned repository snapshot changed: ' + slug)
    tasks_path = root / 'private/tasks.json'
    tasks = json.loads(tasks_path.read_text())['tasks']
    admission = json.loads((root / 'runs/admission/admission.json').read_text())
    ids = [t['id'] for t in tasks]
    if (len(ids) != len(set(ids)) or len(ids) != protocol['execution']['expected_tasks_per_variant']
            or not admission['complete'] or admission['task_spec_sha256'] != sha256(tasks_path)
            or {t['id'] for t in admission['tasks'] if t['admitted']} != set(ids)
            or sha256(root/'runs/admission/admission.json') != protocol['admission_sha256']):
        raise ValueError('All fixed tasks must pass admission')
    runtime = json.loads((root / 'runtime.json').read_text())
    cpu, gpu = str(root / '.venv-agent/bin/python'), runtime['model_python']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    limits = protocol['execution']
    plan = schedule(ids, list(protocol['variants']))
    deadline = time.monotonic() + limits['max_pipeline_seconds']
    out.mkdir(parents=True)
    (out/'private').mkdir(mode=0o700)
    state = {'schema_version':1, 'started_utc':utc(), 'started_epoch':time.time(),
        'deadline_epoch':time.time()+limits['max_pipeline_seconds'], 'complete':False,
        'protocol_sha256':sha256(protocol_path), 'completed_runs':0,
        'schedule':[{'id':task,'variant':arm} for task,arm in plan], 'stages':[]}
    campaigns = {}
    for arm in protocol['variants']:
        campaigns[arm] = {'schema_version':1, 'started_utc':utc(), 'complete':False,
            'identity':{'tasks_sha256':sha256(tasks_path), 'admission_sha256':protocol['admission_sha256'],
                'protocol_sha256':sha256(protocol_path), 'ids':ids, 'variant':arm,
                'policy':protocol['variants'][arm], 'source_files':campaign_sources()}, 'results':[]}
        atomic_json(out/(arm+'.json'), campaigns[arm])

    def stage(name, **extra):
        state.update(stage=name, updated_utc=utc(), **extra)
        atomic_json(out/'controller.json', state)
        print(name, extra.get('completed_runs', state['completed_runs']), flush=True)

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError('Original total time budget exhausted')
        return left

    def run(label, command, timeout):
        start = time.monotonic()
        process = None
        try:
            with (out/'private'/(label+'.log')).open('w') as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
                code = process.wait(timeout=min(timeout, remaining()))
                if code:
                    raise RuntimeError('Controller failed: ' + label)
        finally:
            terminate(process)
        return time.monotonic()-start

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Study interrupted')
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, interrupted)
    monitor = Telemetry(out/'telemetry.csv', 1, 5)
    backend = None
    with single_writer(root/'private/progress.lock'), single_writer(gpu_lock_path()), tempfile.TemporaryDirectory(prefix='oglm017-') as temporary:
        monitor.start()
        try:
            sock = Path(temporary)/'model.sock'
            backend_out = out/'private/backend'
            stage('loading')
            with (out/'private/backend.log').open('w') as log:
                backend = subprocess.Popen([gpu, '-m', 'one_gpu_lm', 'serve', '--model', runtime['model'],
                    '--profile', str(exp/'model-profile.json'), '--socket', str(sock), '--out', str(backend_out)],
                    env=env, stdout=log, stderr=log, start_new_session=True)
            ready_by = time.monotonic() + min(limits['max_backend_start_seconds'], remaining())
            while not (backend_out/'ready.json').exists():
                if backend.poll() is not None:
                    raise RuntimeError('Backend exited before readiness')
                if time.monotonic() > ready_by:
                    raise TimeoutError('Backend readiness exceeded budget')
                time.sleep(1)
            health = request(sock, {'operation':'health'})
            if health['adapter'] or health['profile_sha256'] != sha256(exp/'model-profile.json'):
                raise ValueError('Serving identity must be the unchanged base')
            atomic_json(out/'artifact.json', health)
            stage('loaded_controls')
            duration = run('loaded-controls', [cpu,'-m','one_gpu_lm.evaluation.preflight','--root',str(root),
                '--tasks',str(tasks_path),'--out',str(out/'loaded-controls'),'--clean-only'], 300)
            controls = json.loads((out/'loaded-controls/admission.json').read_text())
            if len(controls['clean']) != protocol['repository_count'] or not all(c['passed'] for c in controls['clean']):
                raise ValueError('GPU-loaded clean controls failed')
            state['stages'].append({'name':'loaded-controls','seconds':duration})
            for task, arm in plan:
                stage('task', current_id=task, current_variant=arm)
                destination = out/'private'/arm/task
                command = [cpu,'-m','one_gpu_lm.evaluation.agent','--root',str(root),'--tasks',str(tasks_path),
                    '--protocol',str(protocol_path),'--socket',str(sock),'--out',str(destination),
                    '--id',task,'--variant',arm]
                seconds = run(arm+'-'+task, command, limits['max_task_seconds'])
                result = json.loads((destination/'result.json').read_text())
                if result['id'] != task or result['variant'] != arm or result['agent_policy'] != protocol['variants'][arm]:
                    raise ValueError('Task result identity mismatch')
                campaigns[arm]['results'].append(result)
                atomic_json(out/(arm+'.json'), campaigns[arm])
                state['stages'].append({'name':'task','id':task,'variant':arm,'seconds':seconds})
                stage('task_finished', completed_runs=state['completed_runs']+1)
            for arm, receipt in campaigns.items():
                receipt.update(complete=True, finished_utc=utc())
                atomic_json(out/(arm+'.json'), receipt)
            stage('completed', complete=True, finished_utc=utc(), finished_epoch=time.time())
        except BaseException as error:
            stage('failed', error_type=type(error).__name__, finished_utc=utc(), finished_epoch=time.time())
            raise
        finally:
            terminate(backend)
            state['telemetry'] = monitor.stop()
            atomic_json(out/'controller.json', state)


if __name__ == '__main__':
    main()
