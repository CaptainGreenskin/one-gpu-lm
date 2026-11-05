"""Export the preselected Vue / seed 101 / early checkpoint from a completed 008 run."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import torch
from runtime import HERE, digest

TRIAL = "vue-101-early"


def export(checkpoint, tokenizer_path, out, provenance):
    if out.exists():
        raise ValueError("Choose a new bundle directory")
    out.parent.mkdir(exist_ok=True, parents=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as staging:
        bundle = Path(staging) / "bundle"
        bundle.mkdir()
        shutil.copyfile(checkpoint, bundle / "weights.pt")
        shutil.copyfile(tokenizer_path, bundle / "tokenizer.json")
        manifest = {"format": "one-gpu-lm-bpe-v1", "config": list(state["config"]), "provenance": provenance,
                    "sha256": {f: digest(bundle / f) for f in ("weights.pt", "tokenizer.json")}}
        (bundle / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        bundle.rename(out)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    seal_sha = digest(args.run / "sealed.json")
    test = json.loads((args.run / "test-scores.json").read_bytes())
    seal = json.loads((args.run / "sealed.json").read_bytes())
    lock = next(r for r in seal["trials"] if r["trial"] == TRIAL)
    ckpt = args.run / TRIAL / "checkpoint.pt"
    if test["seal_sha256"] != seal_sha or TRIAL not in test["trials"] or digest(ckpt) != lock["checkpoint_sha256"]:
        raise ValueError("Source checkpoint disagrees with the evaluated 008 seal")
    tok = HERE.parent / "004-tokenizer-soak/wiki-bpe.json"
    expected = json.loads((HERE.parent / "007-document-curves/prepared-reference.json").read_bytes())
    if digest(tok) != expected["bpe_sha256"]:
        raise ValueError("Source tokenizer changed")
    print(json.dumps(export(ckpt, tok, args.out, {"trial": TRIAL, "seal_sha256": seal_sha}), indent=2))


if __name__ == "__main__":
    main()
