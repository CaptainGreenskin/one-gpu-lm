# 017 supplement · Reference-screened repository replication

**Completed:** both arms pass the same **3/6** tasks, with no improved or regressed pair. Call-limit/no-edit cases fall **2 → 1**, while task-controller time rises **21.3%** in this run. [Results and curves](results/2026-11-11/) retain all 12 outcomes. No accuracy gain was observed on this separate repository; the primary gate remains failed and the default policy stays unchanged. Passing means the final source passes the external grader, not necessarily a normal agent exit.

The original 017 study discovered readable reference implementations for both fresh repositories after launch but before fresh scoring: AVA includes an identical p-map implementation, and the system has an older Blinker package. The original 72 outcomes, denominator, gate and deadline remain intact.

This separate supplement uses six constructed defects in **python-humanize/humanize**, a repository not previously used in this project. Its [plan](plan.json) was declared before source inspection; task selection and admission did not consult any primary model outcomes. The model, progress reminder, tools, native tool-history format, complete context and per-task budgets are exactly those of 017. There are 12 model runs, one per task and arm, in alternating order, after the original run releases the GPU. The separate pipeline limit is one hour, including loading and checks. No scored task is repeated to choose a better result.

The [frozen protocol](protocol.json) records all identities. [Sources and rights](sources.json) pin MIT-licensed release 4.12.3, compatible with the same Python 3.10 test environment. Only a version metadata file normally created by the upstream build hook is added; target implementations and upstream tests remain unchanged before applying private task defects. All upstream number and filesize tests are used; other modules are outside this task family. Six original/broken/restored checks must pass before scoring. The first admission hit an output limit on one failing task; traceback display was shortened equally for both arms, preserving all six defects and tests. [Preparation review](preflight/preparation-review.json).

[Reference screening](preflight/reference-screen.json) runs inside the same agent sandbox. It inventories `/usr`, test packages, Node, the harness mount and the task workspace for other package-name matches, byte-identical selected source files, and package names inside zip/wheel archives. No copies were found. Two entries could not be read by the same sandbox UID: one permission-denied file and one missing symlink target. Their error categories and path hashes are retained. Other scan errors block admission. This is a bounded check, not a proof about all renamed/embedded implementations or base-model pretraining exposure.

This is a **descriptive replication**, not a replacement gate or an opportunity to tune the rule after seeing results. The six tasks share one repository, have visible tests and get one greedy run per arm. Even a favorable result does not erase historical regressions or authorize a new default, deployment or weight release.

For a prepared private root, use the unchanged primary runner with this supplement's protocol, profile and source ledger copied to its recorded configuration paths:

```bash
python experiments/017-edit-progress/run.py \
  --root /your/prepared/supplement-root \
  --out /your/prepared/supplement-root/runs/edit-progress
```

The frozen file hashes and separate output guards must match. Exact model reruns need the private source/task inputs; public results support report reconstruction without publishing those inputs, commands, trajectories, weights or connection details.
