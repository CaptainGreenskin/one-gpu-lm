"""Train a causal, byte-level GPT on two disjoint UTF-8 text files."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, width, heads, layers, length):
        super().__init__()
        self.tokens = nn.Embedding(256, width)
        self.position = nn.Embedding(length, width)
        self.layers = nn.Sequential(*(Block(width, heads) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device)
        h = self.layers(self.tokens(x) + self.position(pos))
        return F.linear(self.norm(h), self.tokens.weight)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny", action="store_true", help="small model for a pipeline check")
    args = parser.parse_args()
    width, heads, layers, length, batch = (32, 2, 2, 16, 4) if args.tiny else (768, 12, 12, 512, 8)
    if args.steps < 1 or min(args.train.stat().st_size, args.val.stat().st_size) <= length:
        parser.error("steps must be positive and each corpus must exceed the sequence length in bytes")
    train_hash, val_hash = digest(args.train), digest(args.val)
    if train_hash == val_hash:
        parser.error("training and validation data must differ")
    if args.device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        parser.error("the GPU configuration requires a CUDA build with BF16 support")
    torch.manual_seed(args.seed)
    model = GPT(width, heads, layers, length).to(args.device)
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            nn.init.normal_(parameter, std=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_rng = torch.Generator().manual_seed(args.seed)
    train = torch.from_file(str(args.train), size=args.train.stat().st_size, dtype=torch.uint8)
    val = torch.from_file(str(args.val), size=args.val.stat().st_size, dtype=torch.uint8)

    def loss_on(data, rng):
        starts = torch.randint(len(data) - length, (batch, 1), generator=rng)
        tokens = data[starts + torch.arange(length + 1)].long().to(args.device)
        precision = torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else nullcontext()
        with precision:
            logits = model(tokens[:, :-1])
            return F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].reshape(-1))

    @torch.no_grad()
    def val_bpb():
        rng = torch.Generator().manual_seed(2026)
        losses = [loss_on(val, rng).item() for _ in range(8)]
        return sum(losses) / (len(losses) * math.log(2))

    args.run_dir.mkdir(parents=True, exist_ok=False)
    code_rev = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    report = {"git_commit": code_rev, "trainer_sha256": digest(Path(__file__)), "seed": args.seed,
              "steps": args.steps, "parameters": sum(p.numel() for p in model.parameters()),
              "width": width, "heads": heads, "layers": layers, "sequence_bytes": length,
              "batch": batch, "device": args.device, "torch": str(torch.__version__),
              "cuda_build": torch.version.cuda, "train_sha256": train_hash, "val_sha256": val_hash,
              "validation": "8 fixed sampled batches; bits per UTF-8 byte",
              "val_bpb_initial": val_bpb()}
    if args.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_on(train, train_rng)
        loss.backward()
        optimizer.step()
    if args.device == "cuda":
        torch.cuda.synchronize()
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    seconds = perf_counter() - started
    tokens = args.steps * batch * length
    report.update(train_wall_seconds=seconds, train_tokens=tokens,
                  train_tokens_per_second=tokens / seconds, val_bpb_final=val_bpb())
    torch.save({"model": model.state_dict(), "config": [width, heads, layers, length]},
               args.run_dir / "model.pt")
    result = json.dumps(report, indent=2)
    (args.run_dir / "metrics.json").write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
