"""Verify private CPU admission and immutable 021 inputs without model scoring."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.evaluation.tasks import file_manifest


def snapshots(root, tasks):
    runtime = json.loads((root/'runtime.json').read_text())
    sources = {name: file_manifest((root/'sources'/name.replace('/', '--')).resolve())
               for name in sorted({t['repository'] for t in tasks})}
    sites = {'historical-default': (root/'.venv-agent/lib/python3.10/site-packages').resolve()}
    sites.update({name: root/path for name, path in runtime.get('python_site_overrides', {}).items()})
    cache = {}; dependencies = {}
    for name, directory in sites.items():
        key = str(directory.resolve())
        if key not in cache:
            cache[key] = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob('*'))
                          if p.is_file() and '__pycache__' not in p.parts}
        dependencies[name] = cache[key]
    for directory in sorted((root/'node-deps').iterdir()):
        if directory.is_dir():
            dependencies['node/'+directory.name] = {str(p.relative_to(directory)): sha256(p)
                for p in sorted(directory.rglob('*')) if p.is_file()}
    return sources, dependencies


def verify(root, initialize=False):
    root = Path(root).resolve(); read = lambda p: json.loads(Path(p).read_text())
    exp = root/'experiments/021-observed-recovery'; plan = read(exp/'preparation-plan.json')
    tasks = read(root/'private/tasks.json')['tasks']; parent = read(root/'private/parent020.json')
    old = Path(parent['root']); old_protocol = read(root/'experiments/020-repair-curriculum/protocol.json')
    if sha256(old/'private/tasks.json') != old_protocol['task_spec_sha256']:
        raise ValueError('Historical task identity changed')
    old_tasks = read(old/'private/tasks.json')['tasks']; by_id = {t['id']: t for t in tasks}
    if len(by_id) != 142 or Counter(t['split'] for t in tasks) != Counter(train=55, dev=15, test=72):
        raise ValueError('All task identities/counts must be distinct and frozen')
    for task in old_tasks:
        expected = dict(task)
        if task['split'] == 'test': expected['evaluation_role'] = 'historical_final_regression_not_selection'
        if by_id.get(task['id']) != expected: raise ValueError('Historical or training task content changed')
    fresh = [t for t in tasks if t['id'] not in {t['id'] for t in old_tasks}]
    old_repos = {t['repository'] for t in old_tasks}
    if (len(fresh) != 24 or len({t['repository'] for t in fresh}) != 4
            or old_repos & {t['repository'] for t in fresh}
            or any(t['split'] != 'test' or t['evaluation_role'] != 'fresh_final_not_selection' for t in fresh)):
        raise ValueError('Fresh test repositories must be disjoint, never training or selection')
    old_admission = read(root/'runs/admission020/admission.json')
    if sha256(root/'runs/admission020/admission.json') != old_protocol['admission_sha256']:
        raise ValueError('Inherited control receipt changed')
    new_admission = read(root/'runs/admission-fresh-draft002/admission.json')
    if (not new_admission['complete'] or new_admission['admitted'] != 24
            or new_admission['task_spec_sha256'] != sha256(root/'private/fresh-tasks.json')
            or len(new_admission['clean']) != 4 or not all(r['passed'] for r in new_admission['clean'])
            or any(not r['admitted'] or r['broken']['returncode'] != 1 or not r['restored']['passed']
                   for r in new_admission['tasks'])):
        raise ValueError('Every fresh task needs an actual failing test and passing restored control')
    screens = read(root/'runs/reference-screen021/screens.json')
    if not screens['complete'] or len(screens['screens']) != 4 or not all(r['screen_passed'] for r in screens['screens']):
        raise ValueError('Four independent reference-exposure screens must pass')
    sources, deps = snapshots(root, tasks)
    if any(sources[name] != manifest for name, manifest in read(old/'source-snapshots.json').items()):
        raise ValueError('Inherited source contents changed')
    if any(deps.get(name) != manifest for name, manifest in read(old/'dependency-snapshots.json').items()):
        raise ValueError('Inherited runtime dependencies changed')
    for source in read(exp/'sources.json')['sources']:
        directory = root/'sources'/source['repository'].replace('/', '--')
        if sha256(directory/source['license_path']) != source['license_sha256']:
            raise ValueError('Source license identity changed')
        for item in source.get('selected_files', []):
            if sha256(directory/item['path']) != item['sha256']: raise ValueError('Reviewed source file changed')
    runtime = read(root/'runtime.json')
    reference = (root/runtime['reference_adapter']).resolve()
    if not reference.is_relative_to(root):
        raise ValueError('Reference adapter must have an isolated local copy')
    if sha256(reference/'adapter_model.safetensors') != plan['reference_adapter']['adapter_model_sha256']:
        raise ValueError('Frozen 020 comparator adapter changed')
    framework = read(root/'framework-verification.json')
    if not framework['all_package_python_files_match_pinned_archive'] or framework['revision'] != old_protocol['agent']['revision']:
        raise ValueError('Pinned agent framework changed')
    merged = dict(old_admission, task_spec_sha256=sha256(root/'private/tasks.json'),
                  clean=old_admission['clean']+new_admission['clean'], tasks=old_admission['tasks']+new_admission['tasks'],
                  admitted=142, complete=True, inherited_admission_sha256=sha256(root/'runs/admission020/admission.json'),
                  new_admission_sha256=sha256(root/'runs/admission-fresh-draft002/admission.json'))
    if len(merged['tasks']) != 142 or not all(t['admitted'] for t in merged['tasks']):
        raise ValueError('No missing admission control is allowed')
    values = {'source-snapshots.json': sources, 'dependency-snapshots.json': deps, 'runs/admission/admission.json': merged}
    for name, value in values.items():
        if initialize:
            if (root/name).exists():
                if read(root/name) != value: raise ValueError('Keep different original preparation receipts')
            else:
                (root/name).parent.mkdir(parents=True, exist_ok=True)
                write_json(root/name, value)
        elif read(root/name) != value: raise ValueError('Frozen preparation receipt changed: '+name)
    result = {'passed': True, 'tasks': 142, 'train': 55, 'development': 15, 'historical': 48, 'fresh': 24,
              'repositories': len(sources), 'new_reference_screens': 4, 'raw_tasks_public': False,
              'task_spec_sha256': sha256(root/'private/tasks.json'), 'admission_sha256': sha256(root/'runs/admission/admission.json')}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--initialize', action='store_true'); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); result = verify(a.root, a.initialize)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, result); print(json.dumps(result))
