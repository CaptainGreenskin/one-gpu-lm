# 016 · Native tool round trips

This is a **historical regression diagnostic**, not a fresh benchmark or weight-training run. All 24 tasks have already been observed in [015](../015-agent-interface/results/2026-11-10/). No task text, mutation, grader or source snapshot changes. The known `new-json-undefined` regression motivated this experiment: its native-schema trajectory repeated read commands and exhausted 12 requests without changing source.

015 supplied a native tool schema but sent observations as ordinary user messages. The pinned model template distinguishes tool responses. Here we compare:

| Policy | Tool definitions and prompts | Executed history |
| --- | --- | --- |
| schema | Same native bash schema and prompts as 015 format | Assistant text + ordinary user observations |
| roundtrip | Identical | Accepted commands become `assistant.tool_calls`; actual observations become `role=tool` |

The same unchanged 27B BF16 model, strict parser, sandbox, 12-call/512-output-per-call/6,144-total-output/8,192-input budgets and full history apply. No increase in budget, anti-repetition hint, answer, semantic summary or automatic command repair. The intervention bundles call and result serialization, so a difference cannot be attributed to one alone. Full original histories remain private and unchanged.

Only an observation with executor metadata and a matching previously accepted action can become a tool response. Invalid-format feedback stays a user message; incomplete or mismatched actions fail validation. Prose, exact command arguments and observation content are preserved. Old policies and default tokenization remain unchanged.

Run all 24 tasks under both arms (48 runs), alternating which policy runs first for each task. All existing sources retain their licenses and stay private. Clean/broken/restored admission and real pinned-tokenizer replay precede scoring. Freeze code, protocol, inputs and controls publicly before any new model rollout. No tuning or repeated best-of attempts after outcomes.

A positive diagnostic must fix the known target, have at least one net gain and zero individual regressions against the concurrent schema control. **It cannot authorize deployment or establish generalization.** Future independent tasks must be frozen before scoring. Compare control outcomes with 015 descriptively to expose repeat-to-repeat variation; do not select the better run.

```bash
python -m unittest tests.test_tool_history tests.test_agent_interface
python experiments/016-tool-roundtrip/run.py --root /your/prepared/root --out runs/roundtrip
```

The runner refuses an existing output. Use the [shared evaluation contract](../../one_gpu_lm/README.md) and freeze new private task/source/dependency identities for your own authorized repositories. Raw task definitions, traces, answers, source snapshots and model weights are not released, so exact score reconstruction needs private inputs; the workflow can be reused with your own inputs.

References: [Transformers tool conversation API](https://huggingface.co/docs/transformers/chat_extras), the pinned local chat template (hash retained in preflight), and [source provenance](sources.json). Original harness code is MIT; upstream source and model rights remain separate.
