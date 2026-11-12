# Experimental adapter decision record

Does not meet the fixed release gates. Keep the base; do not publish candidate weights.

Base: `Qwen/Qwen3.8-27B` at `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.

A LoRA adapter requires this exact base and is not a complete 27B model. No weights are included here. See [the measured outcome and limitations](README.md), [fixed rules](preparation-plan.json), [source provenance](sources.json) and [post-run integrity](postrun-integrity.json).

Scope: experimental repository repair in isolated task sandboxes. One training seed; constructed Python/JavaScript defects; four fresh repository clusters only if final testing opened. No broad coding, long-task or production readiness certification. Training data contains no external teacher API responses or development/final answers. Failed training attempts are deliberately constructed, not sampled model mistakes. Preserve the base and source notices with any authorized adapter distribution.
