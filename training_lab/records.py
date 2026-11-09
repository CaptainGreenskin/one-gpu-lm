"""Versioned input contract and content identities, independent of model architecture."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def load_split(path, split):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = set()
    for row in rows:
        required = {"id", "language", "prompt", "cases", "provenance", "target" if split == "train" else "reference"}
        if not required <= row.keys() or row["id"] in ids or not row["cases"]:
            raise ValueError("Invalid, duplicate, or untestable record")
        if row["language"] not in {"python", "javascript", "cpp"}:
            raise ValueError("Unsupported language")
        if not row["provenance"].get("source") or not row["provenance"].get("kind"):
            raise ValueError("Missing provenance")
        if not row["provenance"].get("rights_basis") or row["provenance"].get("training_allowed") is not True:
            raise ValueError("Missing affirmative training-use rights record")
        ids.add(row["id"])
    if not rows:
        raise ValueError("Empty split")
    return rows


def data_receipt(train_path, test_path, train, test):
    if {r["id"] for r in train} & {r["id"] for r in test}:
        raise ValueError("Train/test IDs overlap")
    if {r["prompt"] for r in train} & {r["prompt"] for r in test}:
        raise ValueError("Exact train/test prompt overlap")
    return {"schema_version": 1, "raw_data_public": False, "splits": {
        name: {"sha256": sha256(path), "records": len(rows),
               "languages": dict(Counter(r["language"] for r in rows)),
               "case_count": sum(len(r["cases"]) for r in rows),
               "provenance_kinds": sorted({r["provenance"]["kind"] for r in rows}),
               "rights_basis": sorted({r["provenance"]["rights_basis"] for r in rows}),
               "redistribution_allowed": all(r["provenance"].get("redistribution_allowed") is True for r in rows)}
        for name, path, rows in [("train", train_path, train), ("test", test_path, test)]}}


def write_json(path, value):
    def encode_special(item):
        if isinstance(item, (set, frozenset)):
            return sorted(item)
        raise TypeError(f"Unsupported receipt type: {type(item).__name__}")
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                    default=encode_special) + "\n")
