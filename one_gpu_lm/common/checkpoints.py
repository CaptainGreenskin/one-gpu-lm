"""Commit complete checkpoints before moving the latest pointer; reject changed inputs/corruption."""
import json
import os
from pathlib import Path
import uuid

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.runstate import atomic_json, identity_digest, sync_directory


def commit_checkpoint(run, step, identity, metadata, writer):
    if {'schema_version','step','identity_sha256'} & metadata.keys():
        raise ValueError('Reserved checkpoint metadata field')
    root=Path(run)/'private/checkpoints';root.mkdir(parents=True,exist_ok=True)
    name=f'step-{step:08d}-'+uuid.uuid4().hex[:12]
    pending=root/('.pending-'+name);pending.mkdir()
    writer(pending)
    if any(p.is_symlink() for p in pending.rglob('*')):
        raise ValueError('Checkpoint writer produced a symbolic link')
    write_json(pending/'trainer.json',{'schema_version':1,'step':step,'identity_sha256':identity_digest(identity),**metadata})
    files={str(p.relative_to(pending)):sha256(p) for p in sorted(pending.rglob('*')) if p.is_file()}
    if not files:
        raise ValueError('Checkpoint writer produced an invalid artifact')
    manifest={'schema_version':1,'step':step,'identity_sha256':identity_digest(identity),'files':files}
    write_json(pending/'manifest.json',manifest)
    for p in pending.rglob('*'):
        if p.is_file():
            with p.open('rb') as stream:os.fsync(stream.fileno())
    for p in sorted((p for p in pending.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        sync_directory(p)
    sync_directory(pending)
    final=root/name;os.replace(pending,final);sync_directory(root)
    pointer={'schema_version':1,'step':step,'checkpoint':str(final.relative_to(run)),
             'manifest_sha256':sha256(final/'manifest.json')}
    atomic_json(Path(run)/'latest.json',pointer)
    return pointer,manifest


def load_checkpoint(run, identity):
    run=Path(run).resolve()
    pointer=json.loads((run/'latest.json').read_text())
    if pointer.get('schema_version')!=1:raise ValueError('Unsupported checkpoint pointer schema')
    relative=Path(pointer['checkpoint'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:2]!=('private','checkpoints'):
        raise ValueError('Unsafe checkpoint pointer')
    root=run/relative
    if root.is_symlink() or not root.resolve().is_relative_to(run/'private/checkpoints'):
        raise ValueError('Checkpoint leaves the run directory')
    if (root/'manifest.json').is_symlink() or sha256(root/'manifest.json')!=pointer['manifest_sha256']:
        raise ValueError('Checkpoint manifest checksum mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('schema_version')!=1:raise ValueError('Unsupported checkpoint manifest schema')
    if manifest['identity_sha256']!=identity_digest(identity) or manifest['step']!=pointer['step']:
        raise ValueError('Checkpoint identity or step changed')
    for name,digest in manifest['files'].items():
        part=Path(name)
        if part.is_absolute() or '..' in part.parts:
            raise ValueError('Unsafe checkpoint member')
        file=root/part
        if file.is_symlink() or not file.resolve().is_relative_to(root) or sha256(file)!=digest:
            raise ValueError('Checkpoint content checksum mismatch')
    trainer=json.loads((root/'trainer.json').read_text())
    if trainer['step']!=pointer['step'] or trainer['identity_sha256']!=identity_digest(identity):
        raise ValueError('Trainer state identity mismatch')
    return root,trainer,manifest
