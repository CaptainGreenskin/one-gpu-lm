"""Short, synthetic capacity benchmark: real 7B updates on one CUDA GPU, no downloaded weights."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "002-first-train"))
from train import GPT


class AdapterQKV(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device="cuda", dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device="cuda", dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["full-adafactor", "lora-adamw"], required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error("choose a new result path")
    free, total = torch.cuda.mem_get_info()
    full = args.mode == "full-adafactor"
    required = 36 if full else 21
    if free < required * 2**30:
        raise SystemExit(f"Insufficient free memory for this probe (requires {required} GiB free)")
    torch.manual_seed(1729)
    torch.cuda.reset_peak_memory_stats()
    config = [4096, 32, 32, 384, 32000, 18048]
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            model = GPT(*config)
    finally:
        torch.set_default_dtype(torch.float32)
    base_parameters = sum(p.numel() for p in model.parameters())
    # Match the initialization scale of the small decoder; no pretrained weights enter this test.
    with torch.no_grad():
        for tensor in model.parameters():
            if tensor.ndim >= 2:
                nn.init.normal_(tensor, std=0.02)
    if full:
        optimizer = torch.optim.Adafactor(model.parameters(), lr=0.01, foreach=False)
        probe = model.layers[0].qkv.weight
    else:
        model.requires_grad_(False)
        for block in model.layers:
            block.qkv = AdapterQKV(block.qkv)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
        probe = model.layers[0].qkv.b
    trainable = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable)
    first = probe.detach().flatten()[:4096].clone()
    seq = 128
    batch = 1
    records = []
    for step in range(4):
        text = torch.randint(config[4], (batch, seq + 1), device="cuda")
        torch.cuda.synchronize()
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(text[:, :-1])
            loss = F.cross_entropy(logits.float().flatten(0, 1), text[:, 1:].flatten())
        loss.backward()
        if any(p.grad is None for p in trainable):
            raise RuntimeError("Missing gradient for a trainable tensor")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite loss")
        records.append({"step": step+1, "seconds": elapsed, "loss": loss.item(), "grad_norm": float(grad_norm)})
    delta = (probe.detach().flatten()[:4096].float() - first.float()).abs().max().item()
    if not delta > 0:
        raise ValueError("No measurable parameter change")
    result = {"mode": args.mode, "config": config, "bootstrap": "random weights, synthetic token IDs",
              "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "base_parameters": base_parameters,
              "trainable_parameters": num_trainable, "trainable_elements_with_gradient": sum(p.grad.numel() for p in trainable),
              "parameter_dtype": str(model.tokens.weight.dtype), "batch": batch, "sequence": seq, "records": records,
              "tokens_per_second_after_warmup": 3 * batch * seq / sum(r["seconds"] for r in records[1:]),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "free_before_bytes": free, "device_memory_bytes": total, "probe_max_abs_update": delta}
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
