"""Train a single prospective policy. Its only data inputs are train and validation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

import torch
from torch.nn import functional as F
from policy import EarlyStop

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "004-tokenizer-soak"))
from trial import GPT, atomic_json, read_rows, score_model, tensors

SPEC = json.loads((HERE / "protocol.json").read_bytes())


def train(data, out, seed, arm):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    m = json.loads((data / "manifest.json").read_bytes())
    bos = m["bpe_vocab"]
    config = [768, 12, 12, m["width_bytes"], bos + 1, 3000]
    model = GPT(*config).cuda()
    for p in model.parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, std=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rows = read_rows(data, "train", "bpe", m)
    val = read_rows(data, "val", "bpe", m) if arm == "early" else None
    rng = torch.Generator().manual_seed(seed)
    state = EarlyStop(SPEC["patience"])
    order = hashlib.sha256()
    curve = []
    cap = 8 if arm == "fixed8" else SPEC["max_epochs"]
    selected = None
    updates = 0.0
    for epoch in range(1, cap + 1):
        torch.cuda.synchronize()
        start = time.monotonic()
        ids = torch.randperm(len(rows), generator=rng)
        order.update(struct.pack(f"<{len(ids)}I", *ids.tolist()))
        for ix in ids.split(8):
            x, y, _ = tensors(rows[ix], bos, "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            last_loss = float(loss.detach())
            if not math.isfinite(last_loss):
                raise ValueError("Non-finite training loss")
        updates += time.monotonic() - start
        row = {"epoch": epoch, "update_seconds": updates, "steps": epoch * math.ceil(len(rows) / 8),
               "train_raw_bytes": epoch * m["splits"]["train"]["raw_bytes"],
               "order_sha256": order.hexdigest(), "last_batch_loss": last_loss}
        stop = False
        save = epoch == cap
        if arm == "early":
            score, _, seconds = score_model(model, val, "cuda", m["splits"]["val"], "bpe", bos)
            model.train()
            save, stop = state.observe(score)
            row.update(val_bpb=score, val_seconds=seconds, stale=state.stale, best_epoch=state.best_epoch)
        if save:
            selected = epoch
            torch.save({"config": config, "model": model.state_dict(), "epoch": epoch},
                       out / "checkpoint.tmp")
            os.replace(out / "checkpoint.tmp", out / "checkpoint.pt")
        curve.append(row)
        atomic_json(out / "curve.json", curve)
        if stop:
            break
    atomic_json(out / "decision.json", {"seed": seed, "policy": arm, "trained_epochs": epoch,
                "selected_epoch": selected, "update_seconds": updates, "parameter_count": sum(p.numel() for p in model.parameters())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy", choices=SPEC["policies"], required=True)
    args = parser.parse_args()
    train(args.data, args.out, args.seed, args.policy)
