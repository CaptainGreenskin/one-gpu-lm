"""Recheck pinned local model content without importing model code or accessing the network."""
import hashlib
import json
from pathlib import Path
import time

from one_gpu_lm.common.receipts import sha256


def verify_model(directory, manifest_path, profile):
    manifest = json.loads(Path(manifest_path).read_text())
    if any(manifest[k]!=profile[k] for k in ['model_id','revision']):
        raise ValueError('Model manifest/profile mismatch')
    downloaded = json.loads((Path(directory)/'download-receipt.json').read_text())
    if any(downloaded[k]!=profile[k] for k in ['model_id','revision']):
        raise ValueError('Downloaded model/profile mismatch')
    entries = manifest['files']+manifest['metadata_files']
    names = [e['name'] for e in entries]
    if len(names)!=len(set(names)):
        raise ValueError('Duplicate model manifest entries')
    tick = time.monotonic()
    for entry in entries:
        name = Path(entry['name'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Unsafe model manifest path')
        file = Path(directory)/name
        if not file.is_file() or file.stat().st_size!=entry['size']:
            raise ValueError('Pinned model file size mismatch')
        if entry.get('sha256'):
            valid = sha256(file)==entry['sha256']
        else:
            data=file.read_bytes()
            valid=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==entry['git_blob_sha1']
        if not valid:
            raise ValueError('Pinned model content mismatch')
    return {'model_id':profile['model_id'],'revision':profile['revision'],
            'manifest_sha256':sha256(manifest_path),'weight_files':len(manifest['files']),
            'metadata_files':len(manifest['metadata_files']),'verified_bytes':sum(e['size'] for e in entries),
            'verification_seconds':time.monotonic()-tick,'content_verified':True}
