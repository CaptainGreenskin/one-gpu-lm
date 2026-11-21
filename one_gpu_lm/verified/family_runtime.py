"""Resolve pinned offline runtime manifests for a development panel."""
import json
from pathlib import Path

from one_gpu_lm.common.receipts import sha256
from .tasks import relative


def runtime_files(root, manifest, families):
    root, manifest = Path(root).resolve(), Path(manifest).resolve()
    families = set(families)
    document = json.loads(manifest.read_bytes())
    if 'runtimes' not in document:
        return {family: manifest for family in families}
    entries = document['runtimes']
    if ('mounts' in document or document.get('schema_version') != 1
            or document.get('final_test_opened') is not False or set(entries) != families):
        raise ValueError('Offline runtime manifest does not identify the full development panel')
    resolved = {}
    for family, entry in entries.items():
        path = (root / relative(entry['path'])).resolve()
        if not path.is_relative_to(root) or sha256(path) != entry['sha256']:
            raise ValueError('A development runtime differs from its frozen receipt')
        resolved[family] = path
    return resolved
