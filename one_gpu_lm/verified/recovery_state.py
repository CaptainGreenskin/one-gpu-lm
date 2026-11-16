"""Bounded, immutable actor-state snapshots for paired training continuations.

This captures file contents, permissions and mtime, including empty directories.
It does not claim to preserve inode/ctime, shell processes or the external clock.
Only call it at a completed tool boundary with no live actor command.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat

MAX_ENTRIES = 4096
MAX_BYTES = 64 * 2**20


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Expected a real actor directory')
    result = {}; total = 0
    for path in [root, *sorted(root.rglob('*'))]:
        info = path.lstat(); name = '.' if path == root else path.relative_to(root).as_posix()
        if len(result) >= MAX_ENTRIES:
            raise ValueError('Actor state exceeds entry bound')
        item = {'mode': stat.S_IMODE(info.st_mode), 'mtime_ns': info.st_mtime_ns}
        if stat.S_ISDIR(info.st_mode):
            item['kind'] = 'directory'
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            total += info.st_size
            if total > MAX_BYTES:
                raise ValueError('Actor state exceeds byte bound')
            item.update(kind='file', bytes=info.st_size, sha256=digest(path.read_bytes()))
        else:
            raise ValueError('Symlinks, hardlinks and special actor files are not serializable')
        result[name] = item
    return result


def validate_manifest(manifest):
    if '.' not in manifest or manifest['.']['kind'] != 'directory' or len(manifest) > MAX_ENTRIES:
        raise ValueError('Invalid actor root inventory')
    total = 0
    for name, item in manifest.items():
        path = PurePosixPath(name)
        if name != '.' and (path.is_absolute() or '..' in path.parts or str(path) != name):
            raise ValueError('Noncanonical snapshot path')
        if item['kind'] not in {'directory', 'file'} or not isinstance(item['mode'], int) or not 0 <= item['mode'] <= 0o7777:
            raise ValueError('Invalid snapshot type or mode')
        if not isinstance(item['mtime_ns'], int):
            raise ValueError('Invalid snapshot timestamp')
        if item['kind'] == 'file':
            if not isinstance(item['bytes'], int) or item['bytes'] < 0:
                raise ValueError('Invalid file size')
            total += item['bytes']
            if len(item['sha256']) != 64 or any(c not in '0123456789abcdef' for c in item['sha256']):
                raise ValueError('Invalid file digest')
        if name != '.' and (str(path.parent) not in manifest or manifest[str(path.parent)]['kind'] != 'directory'):
            raise ValueError('Missing parent directory')
    if total > MAX_BYTES:
        raise ValueError('Actor state exceeds byte bound')


def copy_tree(source, destination, manifest):
    source, destination = Path(source), Path(destination)
    validate_manifest(manifest)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Preserve previous actor workspace')
    if inventory(source) != manifest:
        raise ValueError('Source state differs from the captured manifest')
    destination.mkdir(parents=True)
    for name, item in sorted(manifest.items(), key=lambda pair: (len(PurePosixPath(pair[0]).parts), pair[0])):
        if name == '.':
            continue
        target = destination / name
        if item['kind'] == 'directory':
            target.mkdir()
        else:
            shutil.copyfile(source / name, target)
    for name, item in sorted(manifest.items(), key=lambda pair: len(PurePosixPath(pair[0]).parts), reverse=True):
        target = destination if name == '.' else destination / name
        target.chmod(item['mode'])
        os.utime(target, ns=(item['mtime_ns'], item['mtime_ns']))
    if inventory(source) != manifest or inventory(destination) != manifest:
        raise ValueError('Actor state changed during snapshot copy')


def validate_prefix(prefix):
    if prefix['attempted_calls'] != 3 or len(prefix['steps']) != 3:
        raise ValueError('Recovery requires the registered third-call boundary')
    if prefix.get('unknown_inflight_generation_cost') or prefix['boundary'] != 'completed_step':
        raise ValueError('Do not branch an in-flight or unknown generation')
    if prefix.get('submitted') or prefix.get('terminal_reason'):
        raise ValueError('Do not branch a terminal actor')
    if any(s.get('source_after', s['source_before']) != s['source_before'] for s in prefix['steps']):
        raise ValueError('The recovery pilot requires no prefix source edits')
    if prefix['generated_tokens'] != sum(s['generation']['generated_tokens'] for s in prefix['steps']):
        raise ValueError('Prefix generation accounting changed')
    if len(prefix['messages']) != 8 or [m['role'] for m in prefix['messages']] != ['system', 'user', 'assistant', 'user', 'assistant', 'user', 'assistant', 'user']:
        raise ValueError('Prefix must contain three complete feedback pairs')
    if not math.isfinite(prefix['actor_seconds']) or not 0 <= prefix['actor_seconds'] < 600:
        raise ValueError('Prefix has no remaining actor time')


def capture(workspace, destination, prefix):
    workspace, destination = Path(workspace), Path(destination)
    validate_prefix(prefix)
    if destination.exists():
        raise ValueError('Preserve the previous snapshot')
    manifest = inventory(workspace)
    destination.mkdir(parents=True)
    copy_tree(workspace, destination / 'tree', manifest)
    state_bytes = canonical(prefix)
    (destination / 'prefix.json').write_bytes(state_bytes)
    manifest_bytes = canonical(manifest)
    (destination / 'manifest.json').write_bytes(manifest_bytes)
    receipt = {'schema_version': 1, 'prefix_sha256': digest(state_bytes),
               'manifest_sha256': digest(manifest_bytes), 'entries': len(manifest),
               'file_bytes': sum(v.get('bytes', 0) for v in manifest.values()),
               'preserved': ['contents', 'empty directories', 'permission modes', 'mtime_ns', 'visible history', 'prefix budget counters'],
               'not_preserved': ['inode', 'ctime', 'atime', 'external clock', 'process memory']}
    (destination / 'receipt.json').write_bytes(canonical(receipt))
    return receipt


def restore(snapshot, workspace, expected_receipt):
    snapshot, workspace = Path(snapshot), Path(workspace)
    actual_receipt = json.loads((snapshot / 'receipt.json').read_text())
    if actual_receipt != expected_receipt:
        raise ValueError('Snapshot receipt changed')
    prefix_bytes = (snapshot / 'prefix.json').read_bytes()
    manifest_bytes = (snapshot / 'manifest.json').read_bytes()
    if digest(prefix_bytes) != expected_receipt['prefix_sha256'] or digest(manifest_bytes) != expected_receipt['manifest_sha256']:
        raise ValueError('Snapshot history or manifest changed')
    prefix = json.loads(prefix_bytes); validate_prefix(prefix)
    copy_tree(snapshot / 'tree', workspace, json.loads(manifest_bytes))
    return prefix


def remaining_budget(budget, state):
    calls = state['attempted_calls']; tokens = state['generated_tokens']; seconds = state['actor_seconds']
    if not isinstance(calls, int) or not isinstance(tokens, int) or calls < 0 or tokens < 0 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid consumed budget')
    remaining = {'calls': max(0, budget['max_model_calls'] - calls),
                 'generated_tokens': max(0, budget['max_total_generated_tokens'] - tokens),
                 'actor_seconds': max(0., budget['max_task_seconds'] - seconds)}
    remaining['next_generation_cap'] = min(remaining['generated_tokens'], budget['max_generated_tokens_per_call'])
    return remaining
