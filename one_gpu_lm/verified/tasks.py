"""Materialize reviewed source subsets; keep mutation specifications outside agents."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative(name):
    p = PurePosixPath(name)
    if not isinstance(name, str) or not name or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError('Expected a canonical relative file path')
    if any(part in {'.git', 'node_modules', '__pycache__'} for part in p.parts):
        raise ValueError('Unreviewed history, dependencies or bytecode')
    return p


def files(root):
    root = Path(root)
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('Symlink in source workspace')
        if p.is_file():
            result[str(p.relative_to(root))] = digest(p.read_bytes())
    return result


def validate_source(source, snapshot, purpose):
    if purpose not in {'training', 'development'}:
        raise ValueError('Final tests are not part of this cycle')
    if source['approved_for'].get(purpose) is not True:
        raise ValueError('No explicit approval for this use')
    if source['notice_review_complete'] is not True or source['lineage_review_complete'] is not True:
        raise ValueError('Source review incomplete')
    if source['split'] != purpose or not source['family']:
        raise ValueError('Source split or family differs')
    if len(source['revision']) != 40 or any(c not in '0123456789abcdef' for c in source['revision']):
        raise ValueError('Source must be pinned to a full revision')
    expected = source['files']
    for name in expected:
        relative(name)
    if not expected or not source['notices'] or any(n not in expected for n in source['notices']):
        raise ValueError('Missing source or notice bytes')
    if files(snapshot) != expected:
        raise ValueError('Snapshot differs from the complete approved file inventory')


def validate_split(sources, previous_families=()):
    seen = set()
    families = {'training': set(), 'development': set()}
    for s in sources:
        if s['repository'] in seen:
            raise ValueError('Duplicate repository')
        seen.add(s['repository'])
        if s['split'] not in families:
            raise ValueError('Only training and development sources allowed')
        families[s['split']].update(str(v).casefold() for v in [s['family'],*s.get('related_families',[])])
    if families['training'] & families['development']:
        raise ValueError('Repository family crosses training/development boundary')
    if families['development'] & {str(v).casefold() for v in previous_families}:
        raise ValueError('Development repository family was used previously')
    return families


def prepare(snapshot, destination, source, task, *, broken=True, hidden=False):
    validate_source(source, snapshot, task['split'])
    if task['repository'] != source['repository'] or task['family'] != source['family'] or task['revision'] != source['revision']:
        raise ValueError('Task source identity differs')
    editable = task['editable_files']
    if not editable or len(editable) != len(set(editable)) or any(n not in source['files'] for n in editable):
        raise ValueError('Editable inventory differs')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Preserve previous workspace; use a new destination')
    # All inputs are validated before creating a workspace.
    payload = {n: (Path(snapshot) / n).read_bytes() for n in source['files']}
    changed = set()
    for mutation in task['mutations']:
        name = str(relative(mutation['path']))
        if name not in editable or not mutation['before'] or mutation['before'] == mutation['after']:
            raise ValueError('Invalid mutation target')
        original = payload[name].decode()
        if original.count(mutation['before']) != 1:
            raise ValueError('Mutation anchor must occur exactly once')
        if broken:
            payload[name] = original.replace(mutation['before'], mutation['after']).encode()
        changed.add(name)
    if changed != set(editable):
        raise ValueError('Every editable file must belong to the actual fault')
    checks = dict(task['visible_checks'])
    if hidden:
        if set(checks) & set(task['hidden_checks']):
            raise ValueError('Visible and hidden check paths overlap')
        checks.update(task['hidden_checks'])
    for name, text in checks.items():
        relative(name)
        if name in payload or name in editable:
            raise ValueError('Check overwrites source')
        payload[name] = text.encode()
    destination.mkdir(parents=True)
    for name, data in payload.items():
        p = destination / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return files(destination)


def grade(snapshot, destination, source, task, submitted, before, execute):
    """Copy only allowed source edits into a clean workspace with hidden checks."""
    after = files(submitted)
    editable = task['editable_files']
    protected = [n for n, h in before.items() if n not in editable and after.get(n) != h]
    if protected:
        return {'passed': False, 'reason': 'protected_file_tampering', 'protected_files': protected}
    if any(n not in after for n in editable):
        return {'passed': False, 'reason': 'missing_source'}
    if not any(after[n] != before[n] for n in editable):
        return {'passed': False, 'reason': 'no_source_change'}
    prepare(snapshot, destination, source, task, hidden=True)
    for n in editable:
        shutil.copyfile(Path(submitted) / n, Path(destination) / n)
    value = execute(Path(destination), task['grading_command'])
    return {'passed': value['returncode'] == 0, 'reason': value['reason'], 'returncode': value['returncode'],
            'source_sha256': {n: after[n] for n in editable}, 'ignored_new_files': sorted(set(after) - set(before)),
            'grader_output_sha256': digest(value.get('output', '').encode()), 'seconds': value['seconds']}
