"""Rebuild the historical train/validation sets and separately encode new whole test documents."""
import argparse
from array import array
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile
import unicodedata
from urllib.request import urlopen

from tokenizers import Tokenizer
from run import HERE, REFERENCES, SPEC, digest, write


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rust = load("rust_pages", HERE.parent / "005-technical-bytes/prepare.py")
vue = load("vue_pages", HERE.parent / "007-document-curves/fetch_vue.py")
custom = load("doc_encode", HERE.parent / "007-document-curves/run.py")
encoder = rust.encoder
TOKENIZER = HERE.parent / "004-tokenizer-soak/wiki-bpe.json"


def raw_pieces(data, manifest):
    seen = set()
    width = manifest["width_bytes"] + 1
    for split, meta in manifest["splits"].items():
        path = data / f"{split}-byte.bin"
        if digest(path) != meta["byte_sha256"]:
            raise ValueError("Unexpected train/validation byte file")
        ids = array("H")
        ids.frombytes(path.read_bytes())
        if sys.byteorder != "little":
            ids.byteswap()
        for i in range(0, len(ids), width):
            seen.add(bytes(ids[i+1:i+1+ids[i]]))
    return seen


def test_documents(corpus, archive):
    with tarfile.open(archive) as tar:
        if corpus == "rust":
            for n in SPEC["rust_test_appendices"]:
                prefix = f"book-cn-{rust.REV}/"
                matches = [m for m in tar.getmembers() if m.isfile() and m.name.startswith(prefix) and
                           re.fullmatch(rf"src/appendix-{n}-[^/]+\.md", m.name[len(prefix):])]
                if len(matches) != 1:
                    raise ValueError("Missing or ambiguous Rust appendix")
                raw = tar.extractfile(matches[0]).read()
                yield matches[0].name[len(prefix):], raw, rust.clean(raw.decode())
        else:
            for step in SPEC["vue_test_steps"]:
                name = f"src/tutorial/src/step-{step}/description.md"
                raw = tar.extractfile(f"docs-zh-cn-{vue.REV}/{name}").read()
                text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
                yield name, raw, unicodedata.normalize("NFC", text).strip()


def encode_test(corpus, archive, trainval, out, used_names):
    m = json.loads((trainval / "manifest.json").read_bytes())
    seen = raw_pieces(trainval, m)
    tok = Tokenizer.from_file(str(TOKENIZER))
    if digest(TOKENIZER) != m["bpe_sha256"]:
        raise ValueError("BPE mismatch")
    docs = []
    out.mkdir()
    with (out / "test-bpe.bin").open("wb") as stream:
        for name, raw, text in test_documents(corpus, archive):
            if name in used_names:
                raise ValueError(f"Test path already used in train/validation: {name}")
            rows = nbytes = ntokens = removed = 0
            for piece in encoder.chunks(text):
                if piece in seen:
                    removed += 1
                    continue
                ids = tok.encode(piece.decode(), add_special_tokens=False).ids
                if tok.decode(ids, skip_special_tokens=False).encode() != piece:
                    raise ValueError("Non-reversible BPE encoding")
                encoder.write_row(stream, ids)
                nbytes += len(piece)
                ntokens += len(ids)
                rows += 1
            if not rows:
                raise ValueError(f"Empty test document after overlap exclusion: {name}")
            docs.append({"path": name, "original_sha256": hashlib.sha256(raw).hexdigest(),
                         "rows": rows, "raw_bytes": nbytes, "bpe_tokens": ntokens, "excluded_chunks": removed})
    meta = {k: sum(d[k] for d in docs) for k in ("rows", "raw_bytes", "bpe_tokens", "excluded_chunks")}
    meta["bpe_sha256"] = digest(out / "test-bpe.bin")
    write(out / "manifest.json", {"source_revision": rust.REV if corpus == "rust" else vue.REV,
          "trainval_manifest_sha256": digest(trainval / "manifest.json"), "bpe_sha256": m["bpe_sha256"],
          "width_bytes": m["width_bytes"], "bpe_vocab": m["bpe_vocab"], "splits": {"test": meta}, "documents": docs})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for corpus in SPEC["corpora"]:
        p.add_argument(f"--{corpus}-archive", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("choose a new directory")
    args.out.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        work = Path(tmp).resolve()
        ready = work / "ready"
        for corpus, url in (("rust", rust.ARCHIVE), ("vue", vue.URL)):
            archive = getattr(args, corpus + "_archive")
            if archive is None:
                archive = work / f"{corpus}.tar.gz"
                with urlopen(url, timeout=90) as src, archive.open("wb") as dest:
                    shutil.copyfileobj(src, dest)
            data = ready / "trainval" / corpus
            data.parent.mkdir(exist_ok=True, parents=True)
            if corpus == "rust":
                data.mkdir()
                rust.prepare(archive, data)
                used = json.loads((data / "source-files.json").read_bytes())
            else:
                source = work / "vue-source"
                source.mkdir()
                vue.split_archive(archive, source)
                custom.prepare(source / "train", source / "val", data)
                used = json.loads((source / "source-files.json").read_bytes())["pages"]
            if (data / "manifest.json").read_bytes() != REFERENCES[corpus].read_bytes():
                raise ValueError(f"Train/validation split changed: {corpus}")
            target = ready / "test" / corpus
            target.parent.mkdir(exist_ok=True)
            encode_test(corpus, archive, data, target, {d["path"] for d in used})
        ready.rename(args.out)


if __name__ == "__main__":
    main()
