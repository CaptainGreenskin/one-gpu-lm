"""Measure prefill and decode separately on identical forced token streams; audit logits."""
import argparse
import json
from pathlib import Path
from statistics import median
import time

import torch
from runtime import KVCache, Runtime


@torch.inference_mode()
def parity(runtime, tokens):
    """Compare one full forward with prefill, a multi-token suffix, and single-token decode."""
    ref = runtime.model(tokens).float()
    cache = KVCache(tokens.shape[1])
    n = tokens.shape[1]
    pieces = [tokens[:, :3], tokens[:, 3:9], *tokens[:, 9:].split(1, dim=1)]
    out = torch.cat([runtime.cached(part, cache).float() for part in pieces], dim=1)
    if not torch.isfinite(out).all() or not torch.isfinite(ref).all():
        raise ValueError("Nonfinite inference logits")
    diff = (out - ref).abs()
    return {"per_logit_max_abs": diff.max().item(), "per_logit_mean_abs": diff.mean().item(),
            "argmax_matches": int((out.argmax(-1) == ref.argmax(-1)).sum()), "positions": tokens.numel()}


@torch.inference_mode()
def measure(rt, tokens, prompt, steps, cached):
    batch = tokens.shape[0]
    cache = KVCache(prompt + steps) if cached else None
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    def forward(x):
        return rt.cached(x, cache, last_only=True) if cached else rt.full_last(x)
    forward(tokens[:, :prompt]).argmax(-1)
    torch.cuda.synchronize()
    after_prefill = time.perf_counter()
    for i in range(prompt, prompt+steps):
        forward(tokens[:, i:i+1] if cached else tokens[:, :i+1]).argmax(-1)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-after_prefill
    return {"prefill_ms": (after_prefill-start)*1000, "decode_tokens_s": batch*steps/seconds,
            "ms_per_decode_step": 1000*seconds/steps, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--extended", action="store_true", help="measure the follow-up batch/long-prefix sweep")
    args = p.parse_args()
    if args.out.exists():
        p.error("choose a new output path")
    rt = Runtime.load(args.bundle, "cuda")
    input_ids = rt.tokenizer.encode("为了检查推理性能，我们逐个计算下一个标记。", add_special_tokens=False).ids
    def stream(batch, length):
        seq = [rt.bos] + (input_ids * (1 + length // len(input_ids)))[:length-1]
        return torch.tensor([seq] * batch, device=rt.device)
    checked = stream(2, 42)
    check_bf16 = parity(rt, checked)
    sample_prompt = "在计算属性中"
    cache_output = rt.generate(sample_prompt, 24)
    plain_output = rt.generate(sample_prompt, 24, use_cache=False)
    rows = []
    settings = [(1, 320), (8, 320), (32, 128), (32, 320)] if args.extended else [
        (1, 32), (1, 128), (8, 32), (8, 128)]
    for batch, prompt in settings:
        steps = 32
        tokens = stream(batch, prompt+steps)
        for mode in (False, True):
            measure(rt, tokens, prompt, steps, mode)
        samples = {False: [], True: []}
        for rep in range(5):
            for mode in ((False, True) if rep % 2 == 0 else (True, False)):
                samples[mode].append(measure(rt, tokens, prompt, steps, mode))
        for mode, runs in samples.items():
            rows.append({"batch": batch, "prompt_tokens": prompt, "decode_steps": steps, "cached": mode,
                         "median": {k: median(r[k] for r in runs) for k in runs[0]}, "samples": runs})
    # Reload original FP32 weights to separate cache semantics from BF16 kernel rounding.
    cpu = Runtime.load(args.bundle, "cpu")
    cpu_fp32 = parity(cpu, checked.cpu())
    if cpu_fp32["per_logit_max_abs"] > 0.001 or cpu_fp32["argmax_matches"] != cpu_fp32["positions"]:
        raise ValueError(f"FP32 cache equivalence failed: {cpu_fp32}")
    result = {"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "dtype": "bfloat16",
              "bundle_checkpoint_sha256": json.loads((args.bundle/"bundle.json").read_bytes())["sha256"]["weights.pt"],
              "equivalence": {"cpu_fp32": cpu_fp32, "gpu_bf16": check_bf16,
                  "greedy_24_identical_bf16": cache_output["generated_ids"] == plain_output["generated_ids"]},
              "runs": rows}
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
