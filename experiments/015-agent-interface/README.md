# 015 · Tool format and bounded context

This preregistered 2×2 study tests whether an interface change improves repository repairs with the **unchanged Qwen 27B base**. No training or adapter is involved. The 014 adapter was not promoted.

| Variant | Format contract | Request history |
| --- | --- | --- |
| baseline | Existing fenced-Bash prompt; existing strict parser | Full |
| format | One `bash(command)` tool schema in the pinned native chat template, matching system instructions and error feedback | Full |
| context | Existing format contract | Bounded |
| combined | Native schema contract | Bounded |

The parser remains unchanged for every arm. Incomplete or ambiguous actions are rejected; neither prose nor guessed commands are executed. The format factor combines the schema and matching instructions, so it cannot identify their separate effects. It does not claim constrained decoding.

Bounded context starts above 7,168 actual template tokens. It removes the oldest **complete assistant/observation pairs** toward 6,144 tokens, preserving the system, complete task, newest pair and on-disk workspace. An omission notice is appended to the task. It never cuts a command or invents a summary. The same 8,192 input-token maximum still applies; an oversized pinned core fails. Full private trajectories are retained. Lost history may harm quality, which is why success and regressions must be compared.

The 12 previously observed 014 holdouts are historical diagnostics. Twelve new constructed defects span two independent repositories, ItsDangerous and qs, with six tasks each. Clean/broken/restored controls precede any model score. These are not historical issues or SWE-bench, and unknown base pretraining exposure prevents a contamination-free claim. Tests are visible; there is no additional hidden suite.

All four arms get identical call, output-token, context, tool and wall budgets. For each task, arm order rotates through the four variants; the model stays loaded without a cross-request KV cache. Publish every contrast and the factorial interaction, including regressions and incomplete runs. A net improvement of at least two fresh tasks with no fresh or historical regressions only advances a policy to independent replication. Small task clusters and multiple comparisons limit the result.

## Run with your authorized repositories

Use the task and offline sandbox contract in [evaluation](../../one_gpu_lm/README.md) and prepare an isolated Linux root with model/runtime paths, pinned source snapshots, a private task manifest and dependencies. Run `one_gpu_lm.evaluation.preflight` for clean/broken/restored controls. Pin the model revision, code, task bytes, dependencies and admission receipt in a protocol **before** obtaining scores. The included protocol records this study's identities; replace it with a newly frozen protocol for your own inputs.

```bash
python -m unittest tests.test_agent_interface
python experiments/015-agent-interface/run.py --root /your/prepared/root --out runs/interface
```

The runner rejects an existing output directory, changed frozen inputs or failed controls. Completed task results are durable. Recovery requires inspecting the saved task boundary and preserving the original budgets; this command does not automatically restart a partially completed study.

The generic policy switches also work through `one_gpu_lm.evaluation.campaign --variant NAME` with the corresponding `variants` entry in a frozen protocol. Default entry points retain the previous policy.

Original harness code is MIT. [Pinned source provenance](sources.json) retains separate BSD-3-Clause rights; historical provenance is linked there. Source snapshots, task definitions, answers, traces and model files stay private. Exact reconstruction of these scores requires those private inputs; public hashes alone are insufficient. Users can reproduce the workflow using their own authorized documents or repository tasks.

Implementation reference: [Transformers tool templates](https://huggingface.co/docs/transformers/chat_extras) and the model's pinned local `chat_template.jinja`. No community template patch or floating upstream parser is substituted.
