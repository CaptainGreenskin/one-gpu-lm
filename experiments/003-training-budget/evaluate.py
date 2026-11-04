"""Score each eligible validation byte once within its original Wikipedia article."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from itertools import islice
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/002-first-train"))
from train import GPT, digest


def articles(data_dir):
    metadata = (data_dir / "articles.jsonl").read_text().splitlines()
    with (data_dir / "val.txt").open("rb") as stream:
        for record in map(json.loads, metadata):
            if record["split"] == "val":
                body = stream.read(record["bytes"])
                if len(body) != record["bytes"] or stream.read(2) != b"\n\n":
                    raise ValueError("invalid article boundary")
                yield record["id"], body
        if stream.read(1):
            raise ValueError("unexpected bytes after last article")


def windows(documents, length, stride):
    for page_id, body in documents:
        for target in range(1, len(body), stride):
            end = min(target + stride, len(body))
            start = max(0, end - 1 - length)
            yield page_id, body[start:end], target - start - 1


@torch.inference_mode()
def evaluate(checkpoint, documents, device):
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    width, heads, layers, length = state["config"]
    with torch.device("meta"):
        model = GPT(width, heads, layers, length)
    model.load_state_dict(state["model"], assign=True)
    model.to(device).eval()

    stride = max(1, length // 2)
    sequences = windows(documents, length, stride)
    bits, counts = defaultdict(float), defaultdict(int)
    while group := list(islice(sequences, 16)):
        x = torch.zeros((len(group), length), dtype=torch.long)
        y = torch.full_like(x, -100)
        for row, (page_id, segment, skip) in enumerate(group):
            seq = torch.tensor(list(segment), dtype=torch.long)
            x[row, :len(seq)-1] = seq[:-1]
            y[row, skip:len(seq)-1] = seq[skip+1:]
            counts[page_id] += len(seq) - 1 - skip
        precision = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
        with precision:
            logits = model(x.to(device))
        losses = F.cross_entropy(logits.float().transpose(1, 2), y.to(device), reduction="none", ignore_index=-100)
        row_bits = (losses.sum(dim=1, dtype=torch.float64) / math.log(2)).tolist()
        for (page_id, _, _), value in zip(group, row_bits):
            bits[page_id] += value
    if not counts or not all(math.isfinite(v) for v in bits.values()):
        raise ValueError("empty or non-finite evaluation")
    return {"checkpoint_sha256": digest(checkpoint), "device": device, "context_bytes": length, "stride": stride,
            "target_bytes": sum(counts.values()), "bits_per_byte": sum(bits.values()) / sum(counts.values()),
            "articles": [{"id": k, "targets": n, "bits_per_byte": bits[k] / n} for k, n in counts.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/wiki-zh-first")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, articles(args.data_dir), args.device)
    result["val_sha256"] = digest(args.data_dir / "val.txt")
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"{args.out}: {result['target_bytes']} targets, {result['bits_per_byte']:.6f} bits/byte", flush=True)


if __name__ == "__main__":
    main()
