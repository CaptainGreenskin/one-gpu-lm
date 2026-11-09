# Data card — private pilot fixtures

Purpose: pipeline validation, not representative coding training data. Raw records are not published.

## train

- Records: 18; cases: 38; languages: `{"cpp": 6, "javascript": 6, "python": 6}`.
- SHA-256 of exact JSONL bytes: `6d02e5e2c142c86aafc9daf00057de7037b1fc01d49a1a5bcefe14feaaac640f`.
- Provenance kinds: `project-authored-synthetic`.
- Rights record: Project-authored synthetic fixtures; inputs and reference solutions authored in this task; no external teacher service.
- Redistribution allowed in the input manifest: `False`.

## test

- Records: 9; cases: 33; languages: `{"cpp": 3, "javascript": 3, "python": 3}`.
- SHA-256 of exact JSONL bytes: `65874d057d1b9dd19b07adc95f2b3062c37be9ab13acf0c5fdfe01e3632d8d2b`.
- Provenance kinds: `project-authored-synthetic`.
- Rights record: Project-authored synthetic fixtures; inputs and reference solutions authored in this task; no external teacher service.
- Redistribution allowed in the input manifest: `False`.

Train/test IDs and exact prompts are disjoint. The generic validator does not prove semantic, translation, repository or task-family separation; any such claim requires a separate preparation audit. References were executed before evaluating the model. Existing model exposure to common algorithms cannot be ruled out. See the run-specific audit for authoring and teacher provenance. Synthetic material is not automatically a rights-cleared distillation dataset.

The input contract requires a provenance kind, source description, affirmative training permission and rights basis. These fields record an operator's decision; software cannot determine legal ownership. Future datasets need per-source license/version/attribution records and repository/task-family splits, not just a dataset-level license label.
