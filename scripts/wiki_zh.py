"""Build or verify the fixed Chinese Wikipedia subset for the first GPU run."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import shutil
import tempfile
import unicodedata
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data/wiki-zh-first"
REFERENCE = ROOT / "experiments/002-first-train/wiki-zh-reference.json"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SHARD = "20231101.zh/train-00002-of-00006.parquet"
BASE_URL = "https://huggingface.co/datasets/wikimedia/wikipedia"
SHARD_HASH = "ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f"
TRAIN_DOCS, VAL_DOCS = 2048, 256
FILES = ("train.txt", "val.txt", "articles.jsonl", "ATTRIBUTION.txt")


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def get_shard():
    cache = ROOT / ".cache/wiki"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / SHARD_HASH
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temp:
            tmp = Path(temp.name)
        try:
            with urlopen(f"{BASE_URL}/resolve/{REVISION}/{SHARD}?download=true", timeout=120) as response, tmp.open("wb") as dest:
                shutil.copyfileobj(response, dest)
            if sha256(tmp) != SHARD_HASH:
                raise ValueError("downloaded shard checksum mismatch")
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
    return target


def unique_articles(shard):
    import pyarrow.parquet as parquet

    seen_ids, seen_text = set(), set()
    for batch in parquet.ParquetFile(shard).iter_batches(batch_size=512, columns=["id", "url", "title", "text"]):
        for row in batch.to_pylist():
            text = unicodedata.normalize("NFC", row["text"].strip()).replace("\r\n", "\n").replace("\r", "\n")
            body = text.encode("utf-8")
            if not 1024 <= len(body) <= 32768:
                continue
            normalized_hash = hashlib.sha256(" ".join(text.split()).encode("utf-8")).digest()
            page_id = str(row["id"])
            if page_id in seen_ids or normalized_hash in seen_text:
                continue
            seen_ids.add(page_id)
            seen_text.add(normalized_hash)
            rank = hashlib.sha256(f"one-gpu-lm/wiki-v1/{page_id}".encode()).digest()
            yield rank, page_id, row["title"], row["url"], body


def build(shard, directory):
    if sha256(shard) != SHARD_HASH:
        raise ValueError("wrong Wikipedia shard or checksum")
    selected = heapq.nsmallest(TRAIN_DOCS + VAL_DOCS, unique_articles(shard), key=lambda row: row[:2])
    if len(selected) != TRAIN_DOCS + VAL_DOCS:
        raise ValueError("not enough eligible articles")
    with (directory / "train.txt").open("wb") as train, (directory / "val.txt").open("wb") as val, \
            (directory / "articles.jsonl").open("w", encoding="utf-8", newline="\n") as articles:
        for index, (_, page_id, title, url, body) in enumerate(selected):
            split = "val" if index < VAL_DOCS else "train"
            (val if split == "val" else train).write(body + b"\n\n")
            meta = {"id": page_id, "title": title, "url": url, "split": split,
                    "text_sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
            articles.write(json.dumps(meta, ensure_ascii=False, separators=(",", ":")) + "\n")
    (directory / "ATTRIBUTION.txt").write_bytes((
        "中文维基百科贡献者 / Chinese Wikipedia contributors.\n"
        "逐篇标题、原文链接和贡献者入口见 articles.jsonl；文章历史页可查作者。\n"
        f"来源与快照许可：{BASE_URL}/blob/{REVISION}/README.md\n"
        "语料按 CC BY-SA 3.0 分享：https://creativecommons.org/licenses/by-sa/3.0/\n"
        "数据卡另列 GFDL：https://www.gnu.org/licenses/fdl-1.3.html 。部分条目仅按 CC 许可提供。\n"
        "处理：选取部分页面，规范化 Unicode、换行和首尾空白，用两个 LF 字节分隔文章。\n"
    ).encode("utf-8"))
    manifest = {"dataset": "wikimedia/wikipedia", "revision": REVISION, "shard": SHARD,
                "shard_sha256": SHARD_HASH, "train_articles": TRAIN_DOCS, "val_articles": VAL_DOCS,
                "files": {name: {"bytes": (directory / name).stat().st_size, "sha256": sha256(directory / name)}
                          for name in FILES}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def verify(directory, expected):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest != expected:
        raise ValueError("corpus manifest differs from the version committed in the repository")
    for name in FILES:
        if sha256(directory / name) != expected["files"][name]["sha256"]:
            raise ValueError(f"corpus checksum mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--shard", type=Path, help="use a previously downloaded, checksum-verified shard")
    args = parser.parse_args()
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if args.action == "prepare" and not args.data_dir.exists():
        args.data_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.data_dir.parent) as tmp:
            ready = Path(tmp) / "ready"
            ready.mkdir()
            build(args.shard or get_shard(), ready)
            verify(ready, expected)
            ready.rename(args.data_dir)
    verify(args.data_dir, expected)
    print(f"Verified {TRAIN_DOCS} train and {VAL_DOCS} validation articles in {args.data_dir}.")


if __name__ == "__main__":
    main()
