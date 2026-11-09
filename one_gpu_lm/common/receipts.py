"""Content hashes and public-safe JSON serialization."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    def encode_special(item):
        if isinstance(item, (set, frozenset)):
            return sorted(item)
        raise TypeError(f"Unsupported receipt type: {type(item).__name__}")
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                    default=encode_special) + "\n")
