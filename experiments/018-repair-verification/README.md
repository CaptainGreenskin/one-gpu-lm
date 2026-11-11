# 018 · Repair verification under a fixed budget

Status: rule locked; no model rollouts yet. The two new historical regressions in 017 motivate a generic reminder after editing: inspect valid regression evidence, correct the smallest failing behavior, and recheck current source bytes. See [diagnosis](diagnosis.json) and [locked rule](intervention.json).

Control is the explicit 017 progress policy, not the default unmodified agent. Candidate adds only the verification reminder. Both use the same unchanged 27B BF16 model, 12 calls, 512 generated tokens/call, 6,144 total generated tokens, 8,192 input tokens, full history and tool limits. Prompt overhead counts against the input limit. No model-generated summaries, hidden tests, extra tool executions, forced edits, teacher API or weight updates.

All 24 historical pairs and 12 new pairs will be frozen before scoring. Fresh sources must be unused in prior project evaluations and pass a bounded scan for readable reference copies. Always finish both stages without tuning. Preserve every failure and any incomplete run; do not rerun scored tasks. The final grader, not edit count or an observed test exit status, determines success. The preregistered numerical gate is in [protocol](protocol.json).

The reusable policy is `verification: repair_check_v1` with `progress: edit_checkpoint_v1`, `format: native_roundtrip`, `context: full`. The task supplies its regression command. Only that standalone command (optionally prefixed with `cd /work &&`) can establish fresh verification evidence; pipelines and compound edit/test commands remain unknown. This intentionally conservative recognizer may spend calls rechecking a test that already ran. It cannot prove semantic correctness or that a visible test is complete.

Original harness code is MIT. Upstream source rights remain separate. Publish rules, sanitized metrics, audit hashes and curves; keep task mutations, source snapshots, commands, raw trajectories, weights and connection information private. Exact task replay requires the authorized private inputs.
