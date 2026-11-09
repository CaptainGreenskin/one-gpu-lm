# Model card draft — coding LoRA pipeline pilot

Status: **experimental, local weights only; not a published model release**. This document can accompany a later ModelScope release after its release checklist is satisfied. It is not a download page.

## Identity and intended use

Base: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0); exact revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. Loader: `AutoModelForImageTextToText`; dtype `bfloat16`. Adapter: rank 8, alpha 16; exact targets in `profile.json`. Base provenance and shard hashes are recorded in `download.json`; adapter content hashes in `adapter.json`.

Purpose: validate loading, frozen-backbone updates, isolated code evaluation and adapter round-trip on one GPU. It is not accepted for autonomous repository edits, long-running coding agents, production serving or general coding quality claims. Vision capability was not evaluated.

## Evaluation

Private held-out smoke tasks: base 9/9, adapter 7/9; 2 regressions, 0 improvements. Three languages, small self-authored function-level tasks, one seed. No SWE-bench, long-task, contamination, broader regression or robustness acceptance was completed. Training loss reduction alone is not evidence of better coding ability.

## Rights and redistribution

Original orchestration code is MIT. The base's pinned LICENSE is Apache-2.0; this does not audit upstream training data or warrant non-infringement. Adapter distribution requires the appropriate upstream license/notices and a separate review of data rights and actual packaging. No weight upload is authorized by this card. Private training/test text is not redistributed; see DATA_CARD.md and the run-specific audit for input and teacher provenance.

## Use and release requirements

The adapter must be paired with the exact base revision and tokenizer; it is not a standalone model. See the generic runner for `PeftModel.from_pretrained` loading. Before a public ModelScope version, record the repository owner, release version, exact base/adapter/config hashes, dependency lock, complete load command, license/notice files and a substantially stronger held-out evaluation. Publish a linked GitHub result for each released version. Do not merge weights or change quantization while retaining the same version identifier.
