"""Prepare and execute a registered 034 pair on the pinned 032 workspace."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.verified.rollout import atomic


EXPERIMENT = Path('experiments/034-first-action-ablation')
PRIVATE = Path('private/first-action-ablation')
TRAINER = Path('one_gpu_lm/verified/_historic034_candidate.py')
ARMS = ('control', 'masked')


def read(path):
    return json.loads(path.read_text())


def create(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def prepare(root):
    path = root / EXPERIMENT / 'training-plan.json'
    plan = read(path)
    publication = read(root / PRIVATE / 'publication.json')
    if (publication['plan_sha256'] != sha256(path) or publication['anonymous_files_verified'] is not True
            or publication['ci_conclusion'] != 'success'):
        raise ValueError('Publish and check the exact preregistration before training')
    parent_path = root / plan['parent_protocol']
    parent = read(parent_path)
    if sha256(parent_path) != plan['parent_protocol_sha256']:
        raise ValueError('032 training protocol changed')
    inputs = {**parent['frozen_files'], **plan['implementation_sha256'], plan['parent_protocol']: sha256(parent_path),
              str(EXPERIMENT / 'training-plan.json'): sha256(path)}
    for name, expected in inputs.items():
        if sha256(root / name) != expected:
            raise ValueError(f'Frozen training input differs: {name}')
    generated = root / TRAINER
    if generated.exists() or any((root / 'runs' / f'historic034-{arm}-001').exists() for arm in ARMS):
        raise ValueError('Preserve the first 034 attempt')
    subprocess.run(['patch', '-s', '-F', '0', '-o', str(generated), str(root / plan['original_trainer']),
                    str(root / EXPERIMENT / 'trainer.patch')], cwd=root, check=True)
    if sha256(generated) != plan['generated_trainer_sha256']:
        raise ValueError('Patched trainer is not the registered executable')
    inputs[str(TRAINER)] = sha256(generated)
    spec = read(root / parent['cycle_spec'])
    if (plan['arm_order'] != list(ARMS) or plan['loss_weights'] != {'control': 1, 'masked': 0}
            or plan['optimizer_updates_per_arm'] != parent['registered_optimizer_steps']
            or plan['target_receipt']['records'] != parent['registered_records']):
        raise ValueError('Paired plan does not match the 032 schedule')
    for arm in ARMS:
        directory = PRIVATE / arm
        (root / directory).mkdir(mode=0o700)
        result = {**parent,
                  'stage': '034_registered_first_action_loss_ablation',
                  'run_directory': f'runs/historic034-{arm}-001',
                  'publication_receipt': str(directory / 'publication.json'),
                  'supervisor_receipt': str(directory / 'supervisor.json'),
                  'worker_log': str(directory / 'worker.log'),
                  'frozen_files': inputs,
                  'loss_ablation': {'target_receipt': plan['target_receipt'],
                                    'target_loss_weight': plan['loss_weights'][arm]}}
        protocol = root / directory / 'protocol.json'
        create(protocol, result)
        create(root / result['publication_receipt'], {**publication, 'protocol_sha256': sha256(protocol)})
    return spec['candidate']['maximum_training_pipeline_seconds']


def run(root):
    root = root.resolve()
    state_path = root / PRIVATE / 'controller.json'
    state = {'stage': 'preparing', 'started_utc': utc_now(), 'complete': False, 'arms': {},
             'final_test_opened': False, 'weights_public': False}
    create(state_path, state)
    try:
        limit = prepare(root)
        model_python = read(root / 'private/parent-runtime.json')['model_python']
        for arm in ARMS:
            directory = root / PRIVATE / arm
            protocol = directory / 'protocol.json'
            command = [model_python, '-u', '-m', 'one_gpu_lm.verified._historic034_candidate',
                       '--root', str(root), '--protocol', str(protocol)]
            status = {'started_utc': utc_now(), 'protocol_sha256': sha256(protocol), 'complete': False}
            receipt = directory / 'supervisor.json'
            create(receipt, status)
            state['arms'][arm] = status
            state['stage'] = arm
            atomic(state_path, state)
            code = None
            try:
                with (directory / 'worker.log').open('xb') as log:
                    code = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=limit).returncode
            finally:
                result = root / read(protocol)['run_directory'] / 'result.json'
                status.update(worker_returncode=code, timed_out=code is None, finished_utc=utc_now(),
                              complete=True, candidate_eligible=code == 0 and read(result)['complete'] is True)
                atomic(receipt, status)
                atomic(state_path, state)
            if not status['candidate_eligible']:
                raise RuntimeError('Paired arm failed')
        state.update(stage='complete', complete=True)
    except BaseException as error:
        state.update(stage='failed', error_type=type(error).__name__)
        raise
    finally:
        state['finished_utc'] = utc_now()
        atomic(state_path, state)
    return state


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.root)))
