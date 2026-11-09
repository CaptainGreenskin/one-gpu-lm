# Pilot audit and limitations

This record includes unsuccessful attempts; no attempt was selected using test quality. All actual training updates in the completed attempt use the same predeclared profile and private training split.

- `reference-check`: all JavaScript references failed because the launcher resolved the verified Node runtime under the wrong directory. No model was loaded. The runtime location was corrected.
- `reference-check2`: JavaScript still failed because Node v24.22.2 does not accept `--no-expose-wasm`. Removed that unsupported option; kept an isolated network/PID/mount namespace, `--jitless`, a heap limit and process limits. This was an execution-environment failure, not a model failure.
- `reference-check3`: all 27 reference implementations passed before a model was evaluated. Python, JavaScript and C++ are checked against exact expected outputs, which are not mounted inside the sandbox.
- `pilot-001`: checkpoint loading reached receipt serialization, then failed because Transformers 5 returns sets in loading diagnostics. The deterministic JSON encoder now converts sets to sorted arrays. A regression test covers this interface.
- `pilot-002`: interrupted deliberately before training after a CPU tokenizer preflight revealed that Transformers 5 returns a `BatchEncoding` by default. The runner now explicitly requests `return_dict=False`. The recorded `KeyboardInterrupt` is the operator's controlled stop, not an OOM or numerical failure.
- `pilot-003`: completed run, with strict empty missing/unexpected/mismatched/error key lists. Exact executed source hashes are verified against the Git commit in `code-identity.json`; regenerated report code has its own hash in `artifacts.json`.

No private corpus, generated solution, adapter weight, account or connection address is included in this result bundle. ModelScope was used as transport from the official Qwen repository; all bytes had to match the pinned Hugging Face identity. `download.json` records successful ModelScope transfer plus final checks; earlier stalled Hugging Face download attempts are not included in that duration.

The run uses BF16 full base weights and FP32 LoRA parameters, one seed, rank 8, two epochs, assistant-token-only loss, batch size one and gradient checkpointing. The target sequence limit is 1024, but actual samples are much shorter; this is not an 8K/16K capacity measurement. `causal_conv1d` was not installed, so its reference PyTorch path was used; installed FLA supplies gated-delta kernels. Timing includes compilation costs when first encountered and is not a tuned serving benchmark.

The local developer machine passed ten contract tests; five Linux-only tests were skipped there. The GPU host passed all fifteen contract/pairing/isolation tests, including wrong-answer rejection, JSON type distinctions, no host home mount, a network namespace containing only loopback, and a bounded infinite loop. This does not establish a hardened multi-tenant sandbox. GitHub CI only exercises CPU contracts and compilation on its configured Python versions.

The synthetic fixtures were drafted with this session's coding assistant to test the pipeline. They are not a general-purpose, rights-cleared distillation corpus. Raw fixtures remain private; their hashes are public, which supports identity checking but does not enable independent exact-score reproduction. Others can exercise the same public workflow with their own authorized inputs.

This tiny function-level set was reviewed during development and should henceforth be treated as a regression fixture. Neither a perfect score nor decreasing training loss establishes improvement on unseen repositories or long tasks. Real repository-level tasks, multiple training seeds, a new final holdout, agent/tool-budget controls and broader regression testing remain necessary before recommending a daily-use model or distributing an adapter.

The upstream loader intentionally ignores 15 `mtp.*` tensors. Its empty unexpected-key receipt is after that documented filter, not proof that every checkpoint tensor participates in forward passes. The instantiated full conditional model has 27,356,728,560 parameters; vision was loaded but not evaluated.

A separate 114.35 s diagnostic reloaded the saved adapter and checked both regressed tasks with it enabled and disabled. Enabled: 0/2 tasks passed; disabled: 2/2 passed. All four generated texts exactly matched their original baseline/adapted outputs. This supports reproducibility of this run’s adapter-associated regression; it is not a new final holdout or a general claim about LoRA. No extra training or tuning was done. See `control.json`.
