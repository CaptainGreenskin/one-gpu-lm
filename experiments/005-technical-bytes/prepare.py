"""Prepare chapter-disjoint Chinese Rust Book data for transfer and matched-exposure experiments."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import unicodedata
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent / "004-tokenizer-soak"
spec = importlib.util.spec_from_file_location("chunk_encoder", PREVIOUS / "prepare.py")
encoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoder)

REPO = "rust-lang-cn/book-cn"
REV = "cde74c448e301ce8ac7960a0d3dc879efd83635d"
ARCHIVE = f"https://codeload.github.com/{REPO}/tar.gz/{REV}"
# Select full topics before examining any validation score.
VAL_CHAPTERS = {"04", "10", "15", "20"}
FILES = re.compile(r"src/ch([0-9]{2})-[^/]+\.md")
REFERENCE = HERE / "prepared-reference.json"


def clean(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\{\{#.*?\}\}", "", text, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def prepare(archive, out):
    pages = []
    original = {}
    prefix = f"book-cn-{REV}/"
    with tarfile.open(archive) as tar:
        for item in tar:
            if not item.isfile() or not item.name.startswith(prefix):
                continue
            name = item.name[len(prefix):]
            if name in {"LICENSE-MIT", "LICENSE-APACHE"}:
                (out / name).write_bytes(tar.extractfile(item).read())
            match = FILES.fullmatch(name)
            if match:
                raw = tar.extractfile(item).read()
                text = clean(raw.decode("utf-8"))
                if not re.search(r"[\u4e00-\u9fff]", text):
                    raise ValueError(f"expected Chinese text in {name}")
                pages.append((name, "val" if match.group(1) in VAL_CHAPTERS else "train", text))
                original[name] = hashlib.sha256(raw).hexdigest()
    pages.sort()
    if len(pages)!=94 or not all((out / f).exists() for f in ("LICENSE-MIT","LICENSE-APACHE")):
        raise ValueError("unexpected source snapshot")
    heldout = {b for _, split, text in pages if split == "val" for b in encoder.chunks(text)}
    dropped = 0
    docs = {"train": [], "val": []}
    files = []
    for path, split, text in pages:
        count = 0
        for piece in encoder.chunks(text):
            if split == "train" and piece in heldout:
                dropped += 1
                continue
            docs[split].append((path, piece.decode("utf-8")))
            count += len(piece)
        files.append({"path":path,"split":split,"original_sha256":original[path],"used_bytes":count})
    index = json.dumps(files, ensure_ascii=False, indent=2).encode() + b"\n"
    (out/"source-files.json").write_bytes(index)
    source = {"repository":REPO, "revision":REV, "license":"MIT OR Apache-2.0", "files":len(files),
              "val_chapters":sorted(VAL_CHAPTERS), "exact_train_chunks_matching_validation_removed":dropped,
              "file_index_sha256":hashlib.sha256(index).hexdigest()}
    tok=PREVIOUS/"wiki-bpe.json"
    encoder.build(None, out, tok, source=source, reader=lambda _, split: docs[split])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path)
    p.add_argument("--out", type=Path, default=ROOT/"data/rust-zh-pairs")
    args=p.parse_args()
    if args.out.exists():
        p.error("output already exists")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        tmp=Path(tmp)
        archive=args.archive or tmp/"src.tar.gz"
        if args.archive is None:
            with urlopen(ARCHIVE,timeout=90) as response, archive.open("wb") as f:
                shutil.copyfileobj(response,f)
        draft=tmp/"ready"
        draft.mkdir()
        prepare(archive, draft)
        if (draft/"manifest.json").read_bytes()!=REFERENCE.read_bytes():
            raise ValueError("generated data differs from the frozen reference")
        draft.rename(args.out)
    print((args.out/"manifest.json").read_text())


if __name__ == "__main__":
    main()
