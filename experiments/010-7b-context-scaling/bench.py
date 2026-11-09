"""One isolated trial of the predeclared 7B context/checkpointing matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "002-first-train"))
from train import GPT


class AdapterQKV(nn.Module):
    """Same rank-8, scale-1 QKV update as 009; supports CPU verification too."""
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b)


def forward(model, tokens, recompute):
    hidden = model.tokens(tokens) + model.position(torch.arange(tokens.shape[1], device=tokens.device))
    for block in model.layers:
        hidden = checkpoint(block, hidden, use_reentrant=False) if recompute else block(hidden)
    return F.linear(model.norm(hidden), model.tokens.weight)


def adapt(model):
    model.requires_grad_(False)
    for block in model.layers:
        block.qkv = AdapterQKV(block.qkv)


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def state_bytes(value):
    if isinstance(value, torch.Tensor):
        return tensor_bytes([value])
    if isinstance(value, dict):
        return sum(state_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(state_bytes(v) for v in value)
    return 0


def write(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def trial(args, protocol, result):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 is required")
    free, total = torch.cuda.mem_get_info()
    cap = min(protocol["max_allocated_gib"] * 2**30, free - protocol["reserve_free_gib"] * 2**30)
    result.update(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
                  python=platform.python_version(), free_before_bytes=free, device_memory_bytes=total,
                  allocator_cap_bytes=cap, deterministic_algorithms=False)
    if cap < (36 if args.mode == "full-adafactor" else 21) * 2**30:
        result.update(status="skipped_memory", reason="Not enough free memory for the predeclared minimum")
        return
    torch.cuda.set_per_process_memory_fraction(cap / total)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(protocol["model_seed"])
    config = protocol["model_config"]
    started = time.monotonic()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            model = GPT(*config)
    finally:
        torch.set_default_dtype(torch.float32)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                nn.init.normal_(parameter, std=0.02)
    result["base_parameters"] = sum(p.numel() for p in model.parameters())
    full = args.mode == "full-adafactor"
    if not full:
        adapt(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = (torch.optim.Adafactor(trainable, lr=0.01, foreach=False) if full
                 else torch.optim.AdamW(trainable, lr=3e-4))
    probes = ([model.tokens.weight, model.layers[0].qkv.weight, model.layers[-1].mlp[2].weight]
              if full else [model.layers[0].qkv.b, model.layers[-1].qkv.b])
    initial = [p.detach().flatten()[:4096].float().clone() for p in probes]
    fingerprint = [model.tokens.weight, model.position.weight, model.layers[-1].mlp[2].weight]
    if not full:
        fingerprint += [model.layers[0].qkv.a, model.layers[-1].qkv.a]
    result["initial_probe_sha256"] = hashlib.sha256(b"".join(
        t.detach().flatten()[:4096].float().cpu().numpy().tobytes() for t in fingerprint)).hexdigest()
    result.update(trainable_parameters=sum(p.numel() for p in trainable),
                  parameter_bytes=tensor_bytes(model.parameters()), parameter_dtype="torch.bfloat16",
                  adapter_dtype=None if full else "torch.float32")
    torch.cuda.synchronize()
    result["initialization_seconds"] = time.monotonic() - started
    gen = torch.Generator().manual_seed(protocol["data_seed"])
    token_hash = hashlib.sha256()
    records = result["records"]
    for step in range(protocol["warmup_steps"] + protocol["measured_steps"]):
        ids = torch.randint(config[4], (protocol["batch"], args.sequence + 1), generator=gen)
        token_hash.update(ids.numpy().astype("<i8").tobytes())
        ids = ids.to("cuda")
        torch.cuda.synchronize()
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = forward(model, ids[:, :-1], args.checkpoint)
            loss = F.cross_entropy(logits.float().flatten(0, 1), ids[:, 1:].flatten())
        loss.backward()
        if any(p.grad is None for p in trainable):
            raise RuntimeError("Missing gradient on a trainable tensor")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
        loss_value, norm_value = loss.item(), float(norm)
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite loss")
        records.append(dict(step=step + 1, warmup=step < protocol["warmup_steps"], seconds=seconds,
                            loss=loss_value, grad_norm=norm_value))
        # Do not retain previous logits across the next forward pass.
        del logits, loss, norm, ids
    result["input_sha256"] = token_hash.hexdigest()
    deltas = [(p.detach().flatten()[:4096].float() - old).abs().max().item() for p, old in zip(probes, initial)]
    if not all(delta > 0 for delta in deltas):
        raise ValueError("At least one sampled parameter did not change")
    measured = [r for r in records if not r["warmup"]]
    result.update(status="ok", probe_max_abs_updates=deltas,
                  trainable_elements_with_gradient=sum(p.grad.numel() for p in trainable),
                  gradient_bytes=tensor_bytes(p.grad for p in trainable), optimizer_state_bytes=state_bytes(optimizer.state),
                  tokens_per_second=protocol["batch"] * args.sequence * len(measured) / sum(r["seconds"] for r in measured))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full-adafactor", "lora-adamw"], required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = (HERE / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    if args.out.exists() or args.sequence not in protocol["sequences"] or args.repeat not in range(protocol["repeats"]):
        parser.error("Choose an unused result path and declared sequence/repeat")
    result = dict(mode=args.mode, sequence=args.sequence, checkpoint=args.checkpoint, repeat=args.repeat,
                  batch=protocol["batch"], config=protocol["model_config"], records=[],
                  protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(), scope=protocol["scope"])
    started = time.monotonic()
    try:
        trial(args, protocol, result)
    except torch.cuda.OutOfMemoryError:
        result.update(status="oom", reason="CUDA allocator could not satisfy the requested allocation")
    except Exception as exc:
        # Avoid serializing environment paths or other private exception payloads.
        result.update(status="error", error_type=type(exc).__name__)
        raise
    finally:
        if torch.cuda.is_initialized():
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        result["trial_body_seconds"] = time.monotonic() - started
        write(args.out, result)


if __name__ == "__main__":
    main()
