# 017 · From inspection to an edit

Three historical regressions in [016](../016-tool-roundtrip/results/2026-11-11/) exhausted their 12 calls without attempting an edit. Manual review found one repeated reproduction command and two extended source/test inspection sequences; these were not failed patches. [Sanitized diagnosis](diagnosis.json) preserves the observation and its limits.

The [intervention rule](intervention.json) was locked **before reviewing the new repositories' source**. Both arms use native tool round trips, the unchanged 27B BF16 model, strict action parser, full context, 12 calls, 512 output tokens per call, 6,144 total output tokens and an 8,192 input limit. Only the progress arm receives a temporary system reminder when no allowed source file currently differs from its initial defective state and either four calls have completed or two identical commands have executed consecutively. The reminder reports remaining calls and asks for a justified minimal repair and validation. It contains no reference answer or task-specific hint and does not force an edit. Its tokens count against the same input limit; overflows remain failures.

Current source hashes and progress are measured in both arms. Raw history and tool observations stay unchanged. First source change is a secondary measure, **not task success**; a useless edit can never earn a pass. Deleted or unsafe files do not count as a valid source change. The previous default policy is unchanged.

The fixed sequence is 24 historical task pairs, then 12 new task pairs from Blinker and p-map: **72 runs**. New task admission checks original, defective and restored code before scoring. Freeze inputs, code, dependencies, model and protocol publicly; then run all pairs, including fresh validation even if the historical gate fails. Do not tune between stages or rerun scored tasks to select a better outcome.

Fresh tasks are constructed defects with visible upstream tests on previously unused repositories. They are not real historical issues, secret tests or a broad benchmark. Authors may overlap earlier repositories, and base pretraining exposure is unknown. Original source and task contents remain private; [source/rights ledger](sources.json) identifies pinned MIT snapshots. No weights are trained or published.

```bash
python -m unittest tests.test_edit_progress tests.test_tool_history tests.test_agent_interface
python experiments/017-edit-progress/run.py --root /your/prepared/root --out runs/edit-progress
```

The runner refuses an existing output. Prepare your own authorized tasks using the [shared evaluation workflow](../../one_gpu_lm/README.md), retaining admission and identity receipts. Exact score reconstruction needs private inputs. The preregistered gate requires fresh net gain of at least two, no fresh or historical task regressions, and all three target repairs. Even a pass permits further replication only, not deployment certification.
