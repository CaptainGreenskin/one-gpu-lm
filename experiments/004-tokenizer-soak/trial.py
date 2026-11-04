"""One fixed-time pretraining trial on common raw-text chunks for byte/BPE models."""
import argparse
import hashlib
import struct
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/002-first-train"))
from train import GPT, digest


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_rows(data, split, arm, manifest):
    rows = manifest["splits"][split]["rows"]
    columns = manifest["width_bytes"] + 1
    file = data / f"{split}-{arm}.bin"
    if file.stat().st_size != rows * columns * 2:
        raise ValueError("unexpected token file size")
    return torch.from_file(str(file), size=rows*columns, dtype=torch.uint16).view(rows, columns).long()


def tensors(table, bos, device):
    lengths = table[:, 0]
    width = int(lengths.max())
    y = table[:, 1:width+1].clone()
    x = torch.empty_like(y)
    x[:, 0] = bos
    x[:, 1:] = y[:, :-1]
    y.masked_fill_(torch.arange(width) >= lengths[:, None], -100)
    return x.to(device), y.to(device), int(lengths.sum())


def score_model(model, val, device, val_meta, arm, bos):
    model.eval()
    val_start = time.monotonic()
    nats = 0.
    val_tokens = 0
    with torch.inference_mode():
        for rows in val.split(16):
            x, y, count = tensors(rows, bos, device)
            with torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext():
                logits = model(x)
            losses = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none", ignore_index=-100)
            nats += float(losses.sum(dtype=torch.float64))
            val_tokens += count
    actual_bytes = val_meta["raw_bytes"]
    expected_tokens = actual_bytes if arm == "byte" else val_meta["bpe_tokens"]
    if val_tokens != expected_tokens:
        raise ValueError("incomplete validation coverage")
    score = nats / (math.log(2) * actual_bytes)
    if not math.isfinite(score):
        raise ValueError("non-finite validation loss")
    validation_seconds = time.monotonic()-val_start
    return score, val_tokens, validation_seconds


def batch_ids(size, rng, epochs):
    if epochs is None:
        while True:
            yield torch.randint(size, (8,), generator=rng)
    else:
        for _ in range(epochs):
            yield from torch.randperm(size, generator=rng).split(8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=["byte", "bpe"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--seconds", type=float)
    budget.add_argument("--epochs", type=int, help="read each raw chunk once per epoch, shuffled without replacement")
    parser.add_argument("--tiny-cpu", action="store_true")
    args = parser.parse_args()
    if (args.seconds is not None and args.seconds <= 0) or (args.epochs is not None and args.epochs < 1):
        parser.error("positive training budget required")
    args.out.mkdir(parents=True, exist_ok=False)
    m = json.loads((args.data / "manifest.json").read_text())
    device = "cpu" if args.tiny_cpu else "cuda"
    width = m["width_bytes"]
    bos = 256 if args.arm == "byte" else m["bpe_vocab"]
    torch.manual_seed(args.seed)
    if args.tiny_cpu:
        hid, heads, layers, ffn = 16, 2, 1, 64
    else:
        hid, heads, layers = 768, 12, 12
        ffn = 3072 if args.arm == "byte" else 3000
    model = GPT(hid, heads, layers, width, vocab_size=bos+1, mlp_width=ffn).to(device)
    for p in model.parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, std=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    data = read_rows(args.data, "train", args.arm, m)
    raw_lengths = data[:, 0] if args.arm == "byte" else read_rows(args.data, "train", "byte", m)[:, 0].clone()
    val = read_rows(args.data, "val", args.arm, m)
    rng = torch.Generator().manual_seed(args.seed)
    precision = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    elapsed = 0.
    steps = tokens = raw_bytes = 0
    next_record = 30
    trace = hashlib.sha256()
    for ix in batch_ids(len(data), rng, args.epochs):
        if args.seconds is not None and elapsed >= args.seconds:
            break
        if args.epochs is not None:
            trace.update(struct.pack(f"<{len(ix)}I", *ix.tolist()))
        x, y, count = tensors(data[ix], bos, device)
        optimizer.zero_grad(set_to_none=True)
        with precision():
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        last_loss = float(loss.detach())
        if not math.isfinite(last_loss):
            raise ValueError("non-finite training loss")
        elapsed = time.monotonic() - start
        steps += 1
        tokens += count
        raw_bytes += int(raw_lengths[ix].sum())
        if elapsed >= next_record:
            atomic_json(args.out / "progress.json", {"steps": steps, "elapsed": elapsed, "loss": last_loss,
                                                     "raw_bytes": raw_bytes, "at": time.time()})
            next_record += 30
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    score, val_tokens, validation_seconds = score_model(model, val, device, m["splits"]["val"], args.arm, bos)
    torch.save({"model": model.state_dict(), "arm": args.arm, "bos": bos, "config": [hid, heads, layers, width, bos+1, ffn]},
               args.out / "model.pt")
    metrics = {"arm": args.arm, "seed": args.seed, "requested_seconds": args.seconds, "train_seconds": elapsed,
               "steps": steps, "train_raw_bytes": raw_bytes, "train_tokens": tokens, "train_bytes_per_second": raw_bytes/elapsed,
               "bpb": score, "val_raw_bytes": m["splits"]["val"]["raw_bytes"], "val_tokens": val_tokens, "val_seconds": validation_seconds,
               "parameters": sum(p.numel() for p in model.parameters()), "peak_allocated_bytes": peak, "device": device,
               "manifest_sha256": digest(args.data / "manifest.json"), "torch": str(torch.__version__)}
    if args.epochs is not None:
        if raw_bytes != args.epochs * m["splits"]["train"]["raw_bytes"]:
            raise ValueError("raw-byte exposure mismatch")
        metrics.update(epochs=args.epochs, sample_order_sha256=trace.hexdigest())
    atomic_json(args.out / "metrics.json", metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
