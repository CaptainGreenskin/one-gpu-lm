"""Post-run reference exposure review; never run a model command or change scores."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256, write_json


def command_indicators(messages):
    commands = [a['command'] for m in messages if m.get('role') == 'assistant'
                for a in m.get('extra', {}).get('actions', []) if isinstance(a.get('command'), str)]
    return {
        'accepted_commands': len(commands),
        'explicit_dependency_path_commands': sum(bool(re.search(r'node_modules[/\\]+p-map(?:[/\\\s\x27\x22]|$)', c)) for c in commands),
        'bare_package_import_commands': sum(bool(re.search(r'(?:from\s*|import\s*\(|require\s*\()\s*[\x27\x22]p-map[\x27\x22]', c)) for c in commands),
        'commands_mentioning_node_modules': sum('node_modules' in c for c in commands),
    }


def review(root, run):
    controller = json.loads((run / 'controller.json').read_text())
    assert controller['complete'] and controller['telemetry']['thread_stopped']
    slug = 'sindresorhus--p-map'
    deps = root / 'node-deps' / slug
    reference = deps / 'node_modules/p-map/index.js'
    upstream = root / 'sources' / slug / 'index.js'
    lock = json.loads((deps / 'package-lock.json').read_text())
    tasks = {t['id']: t for t in json.loads((root / 'private/tasks.json').read_text())['tasks']}
    value = {
        'schema_version': 1,
        'review_source_sha256': sha256(Path(__file__)),
        'posthoc': True,
        'scoring_or_inputs_changed': False,
        'repository': 'sindresorhus/p-map',
        'dependency_chain': ['ava@' + lock['packages']['node_modules/ava']['version'],
                             'p-map@' + lock['packages']['node_modules/p-map']['version']],
        'lock_sha256': sha256(deps / 'package-lock.json'),
        'installed_reference_sha256': sha256(reference),
        'target_source_sha256': sha256(upstream),
        'installed_reference_identical_to_target': reference.read_bytes() == upstream.read_bytes(),
        'accessible_to_both_arms': True,
        'reference_read_access_instrumented': False,
        'clean_reference_free_validation': False,
        'interpretation': 'AVA imports its own p-map dependency, readable by the agent. A separate system Blinker package is also visible through /usr. Text indicators cannot prove absence of indirect access, and test execution is not itself evidence that reference source was returned to the model. Keep all frozen scores; neither original fresh repository establishes reference-free generalization.',
        'variants': {},
    }
    reference_text = reference.read_text()
    for arm in ['control', 'progress']:
        campaign = json.loads((run / (arm + '.json')).read_text())
        selected = [r for r in campaign['results'] if r['repository'] == value['repository']]
        assert campaign['complete'] and len(selected) == 6
        rows = []
        for r in selected:
            path = run / 'private' / arm / r['id'] / 'trajectory.json'
            trajectory = json.loads(path.read_text())
            mutations = tasks[r['id']]['mutations']
            rows.append(dict(id=r['id'], trajectory_sha256=sha256(path),
                unmutated_replace_fragments_present_in_reference=[m['before'] in reference_text for m in mutations if m['kind'] == 'replace'],
                **command_indicators(trajectory['messages'])))
        value['variants'][arm] = rows
    system = []
    for pattern in ['lib/python*/dist-packages/blinker', 'lib/python*/site-packages/blinker',
                    'local/lib/python*/dist-packages/blinker', 'local/lib/python*/site-packages/blinker']:
        for path in Path('/usr').glob(pattern):
            system.append({'mounted_path': path.as_posix(), 'base_sha256': sha256(path / 'base.py')})
    value['blinker_system_copies'] = system
    value['blinker_command_indicators'] = {}
    for arm in ['control', 'progress']:
        rows = []
        for r in json.loads((run / (arm + '.json')).read_text())['results']:
            if r['repository'] != 'pallets-eco/blinker':
                continue
            path = run / 'private' / arm / r['id'] / 'trajectory.json'
            messages = json.loads(path.read_text())['messages']
            commands = [a['command'] for m in messages if m.get('role') == 'assistant'
                        for a in m.get('extra', {}).get('actions', []) if 'command' in a]
            rows.append({'id': r['id'], 'trajectory_sha256': sha256(path),
                         'commands_naming_system_blinker_path': sum(bool(re.search(r'/usr/[^\n]*blinker', c)) for c in commands)})
        assert len(rows) == 6
        value['blinker_command_indicators'][arm] = rows
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    assert not a.out.exists()
    write_json(a.out, review(a.root, a.run))
    print('Reference exposure and sanitized command indicators recorded; no scores changed')
