"""Run a real BF16 attention backward pass and report the visible card."""
import json
import os
import sys

import torch
from torch.nn import functional as F


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Check nvidia-smi and the installed PyTorch CUDA build.")
    if torch.cuda.device_count() != 1:
        raise SystemExit("Select one GPU, e.g. CUDA_VISIBLE_DEVICES=0.")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if capability != (12, 0) or "RTX PRO 6000" not in name.upper():
        raise SystemExit(f"Expected RTX PRO 6000 Blackwell (sm_120), got {name}, capability={capability}")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("PyTorch reports that BF16 is unavailable.")
    torch.manual_seed(42)
    x = torch.randn(2, 4, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    attention = F.scaled_dot_product_attention(x, x, x, is_causal=True)
    attention.float().square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(x.grad).all().item():
        raise SystemExit("BF16 attention returned a non-finite gradient.")
    print(json.dumps({"gpu": name, "capability": list(capability), "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                      "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "python": sys.version.split()[0],
                      "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                      "compiled_arches": torch.cuda.get_arch_list(), "bf16_attention_backward": "passed"}, indent=2))


if __name__ == "__main__":
    main()
