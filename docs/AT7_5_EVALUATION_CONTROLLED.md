# AT-7.5 — Controlled Evaluation FP2 Corpus Generation & Readiness Audit

**Living.** Follows AT-7.4 (`EVALUATION_MAPPING_READY`).

**DATA / GENERATION / QUALITY only.** Does **not** train Evaluation.

---

## Verdict

**EVALUATION_CORPUS_READY**

Ready for a later authorized Evaluation path smoke (AT-7.6).  
Does **not** mean `EVALUATION_ADAPTER_READY`.

---

## Corpus

| Item | Value |
|------|-------|
| Version | `fp2-evaluation-controlled-1.0.0` |
| Native count | **252** |
| Scenario families | **63** |
| Record type | `evaluation_judgment` only |
| Provider / dataset_type | `evaluation` |
| Split | `fp2-split-1.0.0` via `metadata.scenario_family` |
| Tree checksum | `764dba2849519c2b3cf1f5ff24acb84c644f3506b99dbc958762e470310e0883` |

### Locations

- `aiodoo-training/fixtures/fp2/reasoning_controlled_1/evaluation/`
- `aiodoo-datasets/datasets/fp2/reasoning_controlled_1/evaluation/`

AT-7.4 `semantics_report.json` preserved alongside natives.

### Files

`evaluation_native.jsonl`, `evaluation_judgment.jsonl`, `pack_evaluation.jsonl`,
`splits.jsonl`, `manifest.json`, `quality_report.json`, `generation_metadata.json`,
`semantics_report.json`

---

## Distributions

### Splits

| Split | Count |
|-------|------:|
| train | 200 |
| validation | 48 |
| test | 4 |
| Family leakage | **0** |

Test is small because family hashing assigned one family to test; isolation held.

### Verdicts

| Verdict | Count |
|---------|------:|
| pass | 126 |
| fail | 63 |
| inconclusive | 63 |

### Score / explanation

| Metric | Count |
|--------|------:|
| with score | 168 |
| without score | 84 |
| with explanation | 177 |
| without explanation | 75 |

### Optional fields

| Pattern | Count |
|---------|------:|
| candidate + expectation + rubric | 63 |
| candidate + expectation | 63 |
| candidate + rubric | 63 |
| candidate only | 63 |

### Candidate categories

| Category | Count |
|----------|------:|
| planner / coding / repair / execution / context / conversation / approval | 32 each |
| generic | 28 |

### Domain

| Domain | Count |
|--------|------:|
| Odoo | 124 |
| Generic | 128 |

Engineering WHAT: **none**.

---

## Quality scorecard

| Gate | Result |
|------|--------|
| Exact duplicates | 0 |
| Normalized duplicates | 0 |
| Forbidden HOW | 0 |
| Taxonomy violations | 0 |
| Negative contamination | 0 |
| Legacy contamination | 0 |
| Family leakage | 0 |
| Pack validity | true |
| provider == dataset_type | true |
| Determinism | PASS |
| Semantic audit | PASS (0 issues) |
| LEGACY_PROJECTION | **NOT PERFORMED** |

---

## Production FP2 loader

`JsonlDatasetSource(validate=True)` loads `pack_evaluation.jsonl` with
`record_format=fp2_training_example` and `dataset_type=evaluation` — all 252
examples identity-load without Protocol V1 formatting.

---

## Immutability

| Artifact | Status |
|----------|--------|
| controlled_batch_2 | `728d9bad…f227` unchanged |
| Conversation 232 | unchanged |
| Approval 162 | unchanged |
| Planner / Context / Coding / Repair / Execution smokes | not modified |

---

## Generator

`generators/evaluation_controlled.py`

- Deterministic themes → 63 families × 4 variants
- `EvaluationJudgmentRecord` only
- Analyze + emit with hard gates

---

## Tests

`tests/unit/test_at75_evaluation_controlled.py`

Plus TR-2…TR-7, AT-7.4, impacted Reasoning tests.

---

## Remaining gaps

1. Evaluation path smoke (AT-7.6)
2. Optional future legacy → FP2 projection phase
3. Conversation / Approval path smokes (independent)

---

## Exact next phase

**AT-7.6 — Controlled Evaluation Skill Adapter Path Smoke**

Isolated smoke subset, local Qwen, tiny max_steps, real LoRA/PEFT, reload,
finite logits, provenance. No full Evaluation train.

---

## STOP

No training. No adapter. No merge. No certification. No legacy projection.
No Runtime/Core changes. No batch_2 edits. No commit/push without authorization.
