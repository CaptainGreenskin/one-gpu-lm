"""Run the original, sealed 022 campaigns only after a verified 039 promotion."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, gpu_lock_path, single_writer, utc
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.adapter_bundle import FILES, verify_adapter
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.serving.client import request
from one_gpu_lm.verified.tasks import relative

from promotion import REFERENCE, authorize, read
from host_restart import RESTARTED, RESTART_STATUS, RESTART_LAUNCH
from recovery import EXP, RESUMED


PRIVATE = Path('private/modelscope-development')
FINAL = Path('runs/q039-final')
SEALED_EXPERIMENT = Path('experiments/022-action-repair')
ARMS = ('final-base', 'final-selected')
SNAPSHOT_CHECK = """
from importlib.util import module_from_spec, spec_from_file_location
from importlib.util import find_spec
from pathlib import Path
import json
from one_gpu_lm.common.receipts import sha256

root = Path.cwd()
spec = spec_from_file_location('frozen_snapshot_check', root/'experiments/021-observed-recovery/preparation.py')
module = module_from_spec(spec)
spec.loader.exec_module(module)
sources = json.loads((root/'source-snapshots.json').read_text())
dependencies = json.loads((root/'dependency-snapshots.json').read_text())
actual = module.snapshots(root,[{'repository':name} for name in sources])
if actual != (sources,dependencies):
    raise ValueError('Sealed source or dependency snapshots changed')
framework = Path(find_spec('minisweagent').submodule_search_locations[0])
expected = json.loads((root/'framework-verification.json').read_text())['python_package_files']
if {str(p.relative_to(framework)):sha256(p) for p in framework.rglob('*.py')} != expected:
    raise ValueError('The installed scoring framework differs from the frozen copy')
"""


def promotion_from_finished_run(root, seal_path, review_path):
    restarted = (root/PRIVATE/RESTART_LAUNCH).is_file() or (root/PRIVATE/RESTART_STATUS).is_file()
    status_path = root/PRIVATE/(RESTART_STATUS if restarted else 'pipeline-resumed.json')
    status = read(status_path) if status_path.is_file() else {}
    if status.get('stage') != 'complete' or status.get('complete') is not True:
        return {'approved':False, 'stage':'development_pending', 'final_test_opened':False}
    run = root/(RESTARTED if restarted else RESUMED)
    controller_path = run/'controller.json'
    audit_path = run/'audit.json'
    controller = read(controller_path)
    audit = read(audit_path)
    if (status['final_test_opened'] is not False or controller['final_test_opened'] is not False
            or controller['stage'] != 'completed' or controller['complete'] is not True
            or not controller['backend_stopped'] or not controller['frozen_inputs_unchanged']
            or audit['controller_sha256'] != sha256(controller_path)
            or audit['started_epoch'] != controller['started_epoch']
            or audit['finished_epoch'] != controller['finished_epoch']
            or len(audit['rows']) != len(controller['runs'])
            or restarted != ('restart' in audit['recovery'])):
        raise ValueError('Development audit differs from its completed controller')
    return authorize(root/EXP/'protocol.json',audit_path,seal_path,review_path)


def verify_sealed_files(root):
    """Read the public protocol and hash bytes. Sealed tasks are never parsed here."""
    original = root/SEALED_EXPERIMENT/'protocol.json'
    if sha256(original) != sha256(REFERENCE/'protocol.json'):
        raise ValueError('The original final-test protocol changed')
    protocol = read(original)
    for name,digest in protocol['frozen_files'].items():
        path = root/relative(name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError('A file in the frozen final environment changed')
    return protocol


def frozen_task_ids(root):
    inventory = read(REFERENCE/'results/2026-11-12/task-inventory.json')
    expected = {r['id']:r['evaluation_role'] for r in inventory if r['split']=='test'}
    tasks = [task for task in read(root/'private/tasks.json')['tasks'] if task['split']=='test']
    admission = read(root/'runs/admission/admission.json')
    admitted = {row['id'] for row in admission['tasks'] if row['admitted']}
    actual = {task['id']:task['evaluation_role'] for task in tasks}
    if not admission['complete'] or len(tasks) != len(actual) or actual != expected or not actual.keys() <= admitted:
        raise ValueError('The exact two sealed repository-test cohorts must be admitted')
    return [task['id'] for task in tasks]


def gpu_processes():
    return {int(pid) for pid in subprocess.check_output(
        ['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).split()}


def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=10)


def run(root, sealed, permission):
    if permission['approved'] is not True or permission['stage'] != 'final_evaluation':
        raise ValueError('Final scoring requires a passed development gate')
    candidate = read(root/EXP/'protocol.json')
    return run_approved(sealed, root/relative(candidate['adapter_directory']), permission)


def run_approved(sealed, adapter, permission, *, adapter_profile=None):
    """Score a promoted checkpoint on the original single-attempt final suite."""
    if permission['approved'] is not True or permission['stage'] != 'final_evaluation':
        raise ValueError('Final scoring requires a passed development gate')
    adapter = Path(adapter)
    if (set(permission['checkpoint_sha256']) != set(FILES)
            or any(not (adapter/name).is_file() or (adapter/name).is_symlink()
                   or sha256(adapter/name) != permission['checkpoint_sha256'][name] for name in FILES)):
        raise ValueError('The promoted adapter changed before final scoring')
    protocol = verify_sealed_files(sealed)
    reference_path = sealed/SEALED_EXPERIMENT/'protocol.json'
    if (permission['reference_release_protocol_sha256'] != sha256(reference_path)
            or any(protocol['frozen_files'][n] != h for n,h in permission['frozen_final_files'].items())):
        raise ValueError('The permitted seal and the executing protocol differ')

    runtime = read(sealed/'runtime.json')
    model = Path(runtime['model'])
    profile_path = sealed/SEALED_EXPERIMENT/'profile-action.json'
    profile = read(profile_path)
    if (profile['model_id'],profile['revision']) != (permission['model_id'],permission['base_revision']):
        raise ValueError('The final serving profile differs from the approved base')
    expected_adapter = profile if adapter_profile is None else adapter_profile
    if (expected_adapter['model_id'],expected_adapter['revision']) != (profile['model_id'],profile['revision']):
        raise ValueError('The reviewed adapter recipe has a different base')
    adapter_receipt = verify_adapter(adapter,expected_adapter,permission['checkpoint_sha256'])
    verify_model(model,sealed/'experiments/011-coding-model-selection/weights-manifest.json',profile)
    env = dict(os.environ)
    env.update(read(sealed/'private/inputs.json'))
    env.update(PYTHONPATH=str(sealed),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    cpu = str(sealed/'.venv-agent/bin/python')
    subprocess.run([cpu,'-c',SNAPSHOT_CHECK],cwd=sealed,env=dict(env,CUDA_VISIBLE_DEVICES=''),
                   capture_output=True,timeout=600,check=True)

    out = sealed/FINAL
    with single_writer(gpu_lock_path()):
        has_final_outputs = any(folder.name.startswith(('final-','legacy-final-','artifact-final-'))
                                for folder in (sealed/'runs').glob('*/*'))
        if out.exists() or has_final_outputs or gpu_processes():
            raise RuntimeError('Preserve the original attempt and wait for an idle GPU')
        out.mkdir(mode=0o700)
        logs = out/'private'
        logs.mkdir(mode=0o700)
        now = time.time()
        state = {'schema_version':1, 'started_epoch':now,
                 'deadline_epoch':now+protocol['budget']['max_gpu_pipeline_wall_seconds'],
                 'final_test_opened':True, 'complete':False, 'optimizer_updates':0, 'weights_public':False,
                 'reference_protocol_sha256':sha256(reference_path), 'campaign_variant':'shared',
                 'execution_snapshots_verified':True,
                 'development_audit_sha256':permission['development_audit_sha256'],
                 'adapter_files_sha256':adapter_receipt['adapter_files_sha256'], 'stages':[]}
        atomic_json(out/'promotion.json',permission)

        def stage(name):
            state.update(stage=name, updated_utc=utc())
            atomic_json(out/'controller.json',state)

        def remaining():
            left = state['deadline_epoch']-time.time()
            if left <= 0:
                raise TimeoutError('The original final-evaluation budget expired')
            return left

        telemetry = Telemetry(out/'telemetry.csv',1)
        stage('opening_seal')
        telemetry.start()
        try:
            ids = frozen_task_ids(sealed)
            if len(ids) != permission['cohorts']['fresh']+permission['cohorts']['historical']:
                raise ValueError('The final cohort does not match the permission')
            for arm in ARMS:
                selected = arm=='final-selected'
                if selected and any(sha256(adapter/name) != permission['checkpoint_sha256'][name] for name in FILES):
                    raise ValueError('The selected adapter changed before loading')
                if gpu_processes():
                    raise RuntimeError('The GPU is occupied by another process')
                with tempfile.TemporaryDirectory(prefix='q039-final-') as folder, (logs/(arm+'-backend.log')).open('xb') as log:
                    socket = Path(folder)/'model.sock'
                    backend_out = logs/(arm+'-backend')
                    command = [runtime['model_python'],'-m','one_gpu_lm','serve','--model',str(model),
                               '--profile',str(profile_path),'--socket',str(socket),'--out',str(backend_out)]
                    if selected:
                        command += ['--adapter',str(adapter)]
                    stage('loading_'+arm)
                    backend = subprocess.Popen(command,cwd=sealed,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        ready = time.monotonic()+min(remaining(),protocol['budget']['max_backend_start_seconds'])
                        while not (backend_out/'ready.json').is_file():
                            if backend.poll() is not None or time.monotonic()>=ready:
                                raise RuntimeError('Final-test backend did not become ready')
                            time.sleep(1)
                        health = request(socket,{'operation':'health'})
                        hashes = {FILES[1]:permission['checkpoint_sha256'][FILES[1]]} if selected else {}
                        if (health['model_id'] != profile['model_id'] or health['revision'] != profile['revision']
                                or health['profile_sha256'] != sha256(profile_path) or health['adapter'] is not selected
                                or health['adapter_sha256'] != hashes or gpu_processes() != {backend.pid}):
                            raise ValueError('The model served during final scoring differs from the permitted artifact')
                        atomic_json(out/('artifact-'+arm+'.json'),health)
                        stages = [
                            (arm, 'campaign', ['--root',str(sealed),'--tasks',str(sealed/'private/tasks.json'),
                              '--admission',str(sealed/'runs/admission/admission.json'),'--protocol',str(reference_path),
                              '--split','test','--variant','shared'],10800),
                            ('legacy-'+arm, 'regression', ['--tasks',str(sealed/'private/legacy-functions.jsonl'),
                              '--node',str(Path(runtime['node_root'])/'bin/node')],1200),
                        ]
                        for label,module,arguments,limit in stages:
                            stage(label)
                            with (logs/(label+'.log')).open('xb') as job_log:
                                job_command = [cpu,'-m','one_gpu_lm.evaluation.'+module,
                                               '--socket',str(socket),'--out',str(out/label),*arguments]
                                job = subprocess.Popen(job_command,cwd=sealed,env=env,stdout=job_log,
                                                       stderr=subprocess.STDOUT,start_new_session=True)
                                try:
                                    if job.wait(timeout=min(limit,remaining())):
                                        raise RuntimeError('Final campaign subprocess failed')
                                finally:
                                    terminate(job)
                            if backend.poll() is not None or gpu_processes() != {backend.pid}:
                                raise RuntimeError('GPU ownership changed during final scoring')
                            state['stages'].append(label)
                        campaign = read(out/arm/'campaign.json')
                        legacy = read(out/('legacy-'+arm)/'result.json')
                        if (not campaign['complete'] or [r['id'] for r in campaign['results']] != ids
                                or not legacy['complete'] or len(legacy['results']) != permission['cohorts']['legacy']):
                            raise ValueError('Final campaign result is incomplete')
                    finally:
                        terminate(backend)
            state.update(complete=True,stage='scored_audit_pending')
        except BaseException as error:
            state.update(stage='failed',error_type=type(error).__name__)
            raise
        finally:
            state.update(finished_epoch=time.time(),telemetry=telemetry.stop())
            stage(state['stage'])
    return {'stage':state['stage'], 'final_test_opened':True, 'weights_public':False}


def interrupted(*_):
    raise KeyboardInterrupt('Final campaign interrupted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('development-root','sealed-root','seal','candidate-review'):
        parser.add_argument('--'+key, required=True, type=Path)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    root,sealed = args.development_root.resolve(),args.sealed_root.resolve()
    permission = promotion_from_finished_run(root,args.seal,args.candidate_review)
    if args.execute and permission['approved']:
        for kind in (signal.SIGTERM,signal.SIGINT):
            signal.signal(kind,interrupted)
    value = run(root,sealed,permission) if args.execute and permission['approved'] else permission
    print(json.dumps({k:value[k] for k in ('stage','final_test_opened')}))


if __name__ == '__main__':
    os.umask(0o077)
    main()
