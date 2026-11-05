"""Prepare user-supplied train/validation documents and measure full-split BPE learning curves."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOKENIZER = HERE.parent / "004-tokenizer-soak"
sys.path.insert(0, str(TOKENIZER))
from prepare import build, chunks


def documents(directory):
    docs = []
    for file in sorted(directory.rglob("*")):
        if file.suffix.lower() not in {".md", ".txt"} or not file.is_file():
            continue
        if not file.resolve().is_relative_to(directory):
            raise ValueError(f"File links outside the input directory: {file.name}")
        name = file.relative_to(directory).as_posix()
        text = unicodedata.normalize("NFC", file.read_text(encoding="utf-8-sig")).strip()
        if text:
            docs.append((name, text))
    if not docs:
        raise ValueError(f"No nonempty UTF-8 .md or .txt documents in {directory}")
    return docs


def prepare(train_dir, val_dir, data):
    docs = {"train": documents(train_dir), "val": documents(val_dir)}
    def fingerprint(text):
        return hashlib.sha256(text.encode()).hexdigest()
    seen = set()
    for split in ("val", "train"):
        for name, text in docs[split]:
            h = fingerprint(text)
            if h in seen:
                raise ValueError(f"Duplicate normalized document in {split}: {name}")
            seen.add(h)
    val_chunks = {chunk for _, text in docs["val"] for chunk in chunks(text)}
    removed = 0
    def rows(_, split):
        nonlocal removed
        for name, text in docs[split]:
            for chunk in chunks(text):
                if split == "train" and chunk in val_chunks:
                    removed += 1
                    continue
                yield name, chunk.decode()
    index = {k: [(name, fingerprint(text)) for name, text in group] for k, group in docs.items()}
    source = {"document_counts": {k: len(group) for k, group in docs.items()},
              "document_index_sha256": fingerprint(json.dumps(index, ensure_ascii=False, sort_keys=True))}
    data.mkdir()
    build(None, data, TOKENIZER / "wiki-bpe.json", source=source, reader=rows)
    manifest_file = data / "manifest.json"
    manifest = json.loads(manifest_file.read_bytes())
    if not manifest["splits"]["train"]["rows"]:
        raise ValueError("No training chunks remain after excluding exact validation matches")
    manifest["source"]["exact_train_chunks_excluded"] = removed
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--val-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="new run directory")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--tiny-cpu", action="store_true", help="exercise the workflow with a small CPU model")
    p.add_argument("--reference", type=Path, help="require a byte-identical prepared manifest")
    args = p.parse_args()
    train, val, out = [d.resolve() for d in (args.train_dir, args.val_dir, args.out)]
    if not train.is_dir() or not val.is_dir() or train.is_relative_to(val) or val.is_relative_to(train):
        p.error("supply two separate, non-overlapping document directories")
    if out.is_relative_to(train) or out.is_relative_to(val):
        p.error("use an output directory outside the input trees")
    if args.epochs < 1 or len(args.seeds) != len(set(args.seeds)):
        p.error("use positive epochs and distinct seeds")
    out.mkdir(parents=True, exist_ok=False)
    manifest = prepare(train, val, out / "encoded")
    if args.reference and manifest.read_bytes() != args.reference.read_bytes():
        raise ValueError("Prepared corpus differs from the frozen reference")
    (out / "manifest.json").write_bytes(manifest.read_bytes())
    protocol = {"seeds": args.seeds, "epochs": args.epochs, "arm": "bpe",
                "device": "cpu" if args.tiny_cpu else "cuda", "tiny_cpu": args.tiny_cpu}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if not args.tiny_cpu:
        with (out / "environment.json").open("w") as stream:
            subprocess.run([sys.executable, str(ROOT / "scripts/check_gpu.py"), "--any-bf16-gpu"],
                           stdout=stream, check=True)
    for seed in args.seeds:
        cmd = [sys.executable, str(TOKENIZER / "trial.py"), "--arm", "bpe", "--seed", str(seed),
               "--epochs", str(args.epochs), "--epoch-curve", "--data", str(manifest.parent),
               "--out", str(out / f"seed-{seed}")]
        if args.tiny_cpu:
            cmd.append("--tiny-cpu")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
