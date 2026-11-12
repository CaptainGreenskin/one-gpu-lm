"""Record private completed training states without changing agent requests.

The standard agent still sees a twelve-call budget. Collection stops after the
eighth completed call, leaving room for a diagnostic, repair, verify and submit.
No evaluation task may enter this collection path. Raw states are never public.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.evaluation import agent
from one_gpu_lm.evaluation.grading import generated_files
from one_gpu_lm.evaluation.tasks import file_manifest


def training_task(tasks, task_id):
    matches = [t for t in tasks if t['id'] == task_id]
    if len(matches) != 1 or matches[0]['split'] != 'train':
        raise ValueError('Collect exactly one admitted training task, never evaluation')
    return matches[0]


def record_state(task, checkpoint, workspace, original, destination):
    if task['split'] != 'train' or checkpoint['identity']['id'] != task['id']:
        raise ValueError('Training state/task identity differs')
    if checkpoint.get('boundary') != 'completed_step':
        raise ValueError('Only completed-step states are usable')
    used = checkpoint['n_calls']
    if not 1 <= used <= 8:
        raise ValueError('Collection call budget changed')
    trajectory = checkpoint['trajectory']
    local = trajectory['local_model']
    if used != len(local['responses']) + len(local.get('backend_errors', [])):
        raise ValueError('Actual request coverage differs from completed-call count')
    tracker = EditableProgress(workspace, original, task['editable_files'])
    state = tracker.snapshot()
    sources = {}
    if not state['invalid_editable_files']:
        for name in task['editable_files']:
            path = Path(workspace) / name
            sources[name] = path.read_text()
        hashes = sorted((name, hashlib.sha256(text.encode()).hexdigest()) for name, text in sources.items())
        if hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest() != state['editable_state_sha256']:
            raise ValueError('Source bytes changed while capturing state')
    messages = trajectory['messages']
    try:
        observed_files = file_manifest(workspace, exclude=generated_files(task.get('repository', '')))
        protected_ok = (set(observed_files) == set(original)
                        and all(observed_files[name] == digest for name, digest in original.items()
                                if name not in task['editable_files']))
    except ValueError:
        protected_ok = False
    value = {'schema_version': 1, 'task_id': task['id'], 'split': 'train', 'completed_calls': used,
             'identity': checkpoint['identity'], 'trajectory': trajectory, 'source_state': state,
             'sources': sources, 'terminal': bool(messages and messages[-1]['role'] == 'exit'),
             'protected_files_unchanged': protected_ok,
             'raw_private_evidence': True}
    destination = Path(destination)
    if destination.exists():
        if json.loads(destination.read_text()) != value:
            raise ValueError('Do not overwrite a different recorded state')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_json(destination, value)
        destination.chmod(0o600)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'tasks', 'admission', 'protocol', 'socket', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--id', required=True)
    args = parser.parse_args()
    task = training_task(json.loads(args.tasks.read_text())['tasks'], args.id)
    admission = json.loads(args.admission.read_text())
    if (admission.get('complete') is not True or admission['task_spec_sha256'] != sha256(args.tasks)
            or not any(r['id'] == args.id and r['admitted'] and r['split'] == 'train' for r in admission['tasks'])):
        raise ValueError('Training collection requires exact admitted inputs')
    protocol = json.loads(args.protocol.read_text())
    if protocol['agent']['max_model_calls'] != 12 or protocol['collection']['stop_after_completed_calls'] != 8:
        raise ValueError('Frozen collection/inference call bounds changed')
    out = args.out.resolve()
    if out.exists():
        raise ValueError('Preserve every collection attempt; no automatic score reruns')
    out.mkdir(parents=True, mode=0o700)
    original_writer = agent.atomic_json

    def capture(path, value):
        original_writer(path, value)
        if value.get('boundary') == 'completed_step':
            original = json.loads((out / 'original.json').read_text())
            record_state(task, value, out / 'workspace', original,
                         out / 'states' / ('call-%02d.json' % value['n_calls']))

    agent.atomic_json = capture
    sys.argv = ['agent', '--root', str(args.root), '--tasks', str(args.tasks),
                '--protocol', str(args.protocol), '--socket', str(args.socket),
                '--out', str(out), '--id', args.id, '--variant', 'shared', '--interrupt-after', '8']
    try:
        agent.main()
    finally:
        agent.atomic_json = original_writer


if __name__ == '__main__':
    main()
