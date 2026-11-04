"""Build exactly reversible training/validation chunks for equal raw-byte context comparisons."""
from array import array
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from wiki_zh import REFERENCE, verify, sha256

WIDTH = 384
DEFAULT_OUT = ROOT / "data/tokenizer-pairs"


def documents(data, split):
    metadata = (data / "articles.jsonl").read_text().splitlines()
    with (data / (split + ".txt")).open("rb") as stream:
        for line in metadata:
            row = json.loads(line)
            if row["split"] == split:
                body = stream.read(row["bytes"])
                if hashlib.sha256(body).hexdigest() != row["text_sha256"] or stream.read(2) != b"\n\n":
                    raise ValueError("article boundary or hash mismatch")
                yield row["id"], body.decode("utf-8")
        if stream.read(1):
            raise ValueError("unclaimed article bytes")


def chunks(text):
    # Keep complete UTF-8 characters and reset context at identical boundaries for both arms.
    raw = text.encode("utf-8")
    start = 0
    while start < len(raw):
        end = min(start + WIDTH, len(raw))
        while end < len(raw) and raw[end] & 0xC0 == 0x80:
            end -= 1
        yield raw[start:end]
        start = end


def train_tokenizer(data, path):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=2048, min_frequency=2,
                                  initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()), show_progress=False)
    tok.train_from_iterator((piece.decode() for _, doc in documents(data, "train") for piece in chunks(doc)), trainer)
    tok.save(str(path))


def write_row(file, tokens):
    if not 0 < len(tokens) <= WIDTH:
        raise ValueError("invalid encoded length")
    row = array("H", [len(tokens), *tokens, *([0] * (WIDTH-len(tokens)))])
    if sys.byteorder != "little":
        row.byteswap()
    file.write(row.tobytes())


def build(data, out, tokenizer_file, *, source=None, reader=documents):
    tok = Tokenizer.from_file(str(tokenizer_file))
    record = {"source": json.loads(REFERENCE.read_text()) if source is None else source, "width_bytes": WIDTH, "bpe_vocab": tok.get_vocab_size(),
              "bpe_sha256": sha256(tokenizer_file), "splits": {}}
    for split in ("train", "val"):
        rows = byte_count = bpe_count = 0
        with (out / f"{split}-byte.bin").open("wb") as byte_out, (out / f"{split}-bpe.bin").open("wb") as bpe_out:
            for _, text in reader(data, split):
                for piece in chunks(text):
                    text_piece = piece.decode()
                    ids = tok.encode(text_piece, add_special_tokens=False).ids
                    if tok.decode(ids, skip_special_tokens=False) != text_piece:
                        raise ValueError("tokenization failed to round-trip")
                    write_row(byte_out, list(piece))
                    write_row(bpe_out, ids)
                    rows += 1
                    byte_count += len(piece)
                    bpe_count += len(ids)
        record["splits"][split] = {"rows": rows, "raw_bytes": byte_count, "bpe_tokens": bpe_count,
                                   "byte_sha256": sha256(out / f"{split}-byte.bin"),
                                   "bpe_sha256": sha256(out / f"{split}-bpe.bin")}
    (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/wiki-zh-first")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fit", action="store_true", help="create a fresh vocabulary if none exists")
    args = parser.parse_args()
    verify(args.data, json.loads(REFERENCE.read_text()))
    tokenizer_file = HERE / "wiki-bpe.json"
    if args.fit:
        if tokenizer_file.exists():
            parser.error("refusing to replace the frozen vocabulary")
        train_tokenizer(args.data, tokenizer_file)
    if args.out.exists():
        parser.error("choose a new output directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        draft = Path(tmp) / "data"
        draft.mkdir()
        build(args.data, draft, tokenizer_file)
        draft.rename(args.out)
    print((args.out / "manifest.json").read_text())


if __name__ == "__main__":
    main()
