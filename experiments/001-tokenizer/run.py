# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers==0.21.4"]
# ///
"""Count content tokens on a fixed set of original examples; no model weights needed."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
MODELS = {
    "openai-community/gpt2": (
        "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    ),
    "Qwen/Qwen2.5-0.5B": (
        "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
}


def load_tokenizer(cache_dir, repo, revision, digest):
    path = cache_dir / (repo.split("/")[-1] + ".json")
    if path.exists():
        data = path.read_bytes()
    else:
        with urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/tokenizer.json", timeout=90) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{path} has an unexpected checksum")
    if not path.exists():
        path.write_bytes(data)
    return Tokenizer.from_str(data.decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=HERE.parents[1] / ".cache")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    raw = (HERE / "samples.jsonl").read_bytes()
    samples = [json.loads(line) for line in raw.splitlines()]
    result = {"samples_sha256": hashlib.sha256(raw).hexdigest(), "models": {}}
    for repo, (revision, digest) in MODELS.items():
        tokenizer = load_tokenizer(args.cache_dir, repo, revision, digest)
        counts = {}
        groups = {}
        for sample in samples:
            tokens = len(tokenizer.encode(sample["text"], add_special_tokens=False).ids)
            counts[sample["id"]] = tokens
            group = groups.setdefault(sample["group"], {"texts": 0, "utf8_bytes": 0, "tokens": 0})
            group["texts"] += 1
            group["utf8_bytes"] += len(sample["text"].encode("utf-8"))
            group["tokens"] += tokens
        result["models"][repo] = {"revision": revision, "tokenizer_sha256": digest,
                                  "group_totals": groups, "tokens_by_id": counts}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
