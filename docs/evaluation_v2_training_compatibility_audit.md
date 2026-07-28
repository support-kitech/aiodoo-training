# Evaluation Capability Migration Audit — HISTORICAL

> **Status (2026-07-28):** Superseded. Phases 1–4 of Evaluation contract
> adoption / formatter / validation / production finalization are complete.
> Training now uses judgment SFT via `project_evaluation` +
> `_ContractFormatter`. Keep this document only as the pre-migration audit
> snapshot. Current architecture: `CONTRACT_ADOPTION.md`,
> `PRODUCTION_TRAINING.md`, `configs/training/evaluation/`.

**Original audit date:** 2026-07-28  
**Repository:** `aiodoo-training`  
**Original scope:** Architecture & compatibility audit only (no code changes)  
**Dataset baseline:** Production-certified Evaluation v2 (`evaluation_dataset.jsonl`, 189,615 judgments) + separate BenchmarkCatalog

---

## Original final decision (pre-migration)

### B) Evaluation training pipeline is NOT compatible.

*(Resolved by Phases 1–4. Do not treat this conclusion as current status.)*

List of required implementation tasks is in **§10**. Do not begin implementation in this phase.

---

## 1. Current Evaluation training architecture

Two different “evaluation” concepts coexist in `aiodoo-training`. This audit concerns **(A)** only.

| Concept | Role | Compatible with Evaluation v2 SFT? |
|---------|------|------------------------------------|
| **(A) Evaluation capability LoRA** | Train `aiodoo-evaluation` from `evaluation_dataset.jsonl` via `DatasetType.EVALUATION` + `EvaluationFormatter` | **No** — still BenchmarkCatalog-shaped |
| **(B) Phase 4 training evaluation** | Post-train quality gate (`evaluation.yaml`, `EvaluationEngine`, metrics/loss) | Orthogonal — not the Evaluation capability dataset |

Capability training flow (intended for all contract-backed capabilities):

```text
JSONL record
  → DatasetValidator (required fields + optional contract sample)
  → Formatter (project → CapabilityPromptBuilder → TrainingExample)
  → tokenizer / packing / curriculum
  → LoRA trainer
  → export aiodoo-evaluation
```

For Evaluation today, the formatter **skips** contract projection and hand-builds a catalog prompt (legacy Phase-1 path). Six other capabilities (planner/coding/repair/execution/conversation/approval) already use `_ContractFormatter` → `prompt_bridge.build_training_example`.

`aiodoo-contract` already defines `EvaluationRequest` / `EvaluationResponse` and `CapabilityPromptBuilder` has `_extract_evaluation`. `aiodoo-datasets` already ships `project_evaluation`. **Training never calls either for SFT.**

---

## 2. Files involved

### Core (blocking)

| Path | Role |
|------|------|
| `aiodoo_training/datasets/formatters/formatters.py` | `EvaluationFormatter` — catalog user text; assistant = `{evaluation_id}` |
| `aiodoo_training/contract/adapters.py` | `_PROJECTORS` omits `evaluation`; docs say BenchmarkCatalog gap |
| `aiodoo_training/datasets/validation.py` | `REQUIRED_FIELDS[EVALUATION] = {evaluation_id, metadata}` |
| `aiodoo_training/datasets/formatters/__init__.py` | Registers `evaluation` → `EvaluationFormatter` |
| `aiodoo_training/contract/prompt_bridge.py` | Standard R/R → system/user/assistant (unused by Evaluation) |
| `aiodoo_training/datasets/source.py` | Loads refs through validator + formatter |
| `aiodoo_training/datasets/mixing.py` | `stable_example_id` looks for `evaluation_id`, not `record_id` |

### Config / curriculum

| Path | Role |
|------|------|
| `configs/training/evaluation/dataset.yaml` | Points at `evaluation_dataset.jsonl`; `record_count: 1` (stale) |
| `configs/training/evaluation/experiment.yaml` | Root include; comments say 1 record |
| `configs/training/evaluation/training.yaml` | Curriculum `stages: [evaluation]` |
| `configs/training/evaluation/evaluation.yaml` | Phase-4 quality gate; smoke prompts still catalog-worded |
| `configs/training/evaluation/README.md` | Documents 1-record placeholder |

### Tests / fixtures (encode old shape)

| Path | Role |
|------|------|
| `tests/fixtures/datasets/evaluation.jsonl` | Single BenchmarkCatalog row (`evaluation_id` + `catalog`) |
| `tests/unit/test_formatters.py` | Evaluation in `NON_CONTRACT_CASES` (2-message) |
| `tests/contract/test_dataset_contracts.py` | Required `{evaluation_id, metadata}`, `has_contract_projection=False` |

### Docs (stale vs datasets)

| Path | Role |
|------|------|
| `CONTRACT_ADOPTION.md` §7 | Explicitly: evaluation dataset = BenchmarkCatalog; no projection |
| `PRODUCTION_TRAINING.md` | Evaluation = 1 placeholder record; not training-scale |
| `docs/capability_model.md` | Maps evaluation → `evaluation_dataset.jsonl` (filename OK; grain outdated) |

### Related but out of SFT path

Phase-4: `aiodoo_training/evaluation/*`, `builders/evaluation_builders.py`, `config/evaluation_config.py`, `evaluate.py`, `docs/phase4-evaluation-export-architecture.md`, ADR-0015.

---

## 3. Current data flow

```text
configs/training/evaluation/dataset.yaml
  datasets[0].path = evaluation_dataset.jsonl
  dataset_type = evaluation
        │
        ▼
JsonlDatasetSource (validate=True by default)
        │
        ├─ DatasetValidator.validate_ref
        │     requires: evaluation_id, metadata
        │     contract sample: SKIPPED (evaluation ∉ SUPPORTED_CAPABILITIES)
        │
        ├─ EvaluationFormatter._format
        │     user = "Evaluate using the following catalog:\n" + json(catalog)
        │     assistant = {"evaluation_id": ...}
        │     (no CapabilityPromptBuilder; no EvaluationRequest/Response)
        │
        ▼
TrainingExample (user + assistant only; no system turn)
        │
        ▼
tokenization / packing / sequential curriculum stage "evaluation"
        │
        ▼
publish models/adapters/aiodoo-evaluation
```

**v2 SFT record shape (actual certified data):**

```text
record_id, candidate_id, evaluation_case_key, capability_under_test,
candidate, expectation, rubric, verdict, score, explanation, metadata
```

No `evaluation_id`, no `catalog`.

---

## 4. Current formatter behavior

`EvaluationFormatter` (today):

1. Reads `record["catalog"]`.
2. User message: dump entire catalog JSON under a fixed English prefix.
3. Assistant message: `{"evaluation_id": record.get("evaluation_id")}`.
4. Does **not** project to `EvaluationRequest`/`EvaluationResponse`.
5. Does **not** teach verdict/score/explanation labels.
6. Emits **2** messages (user/assistant), unlike contract formatters (system/user/assistant via ADR-0003).

Target behavior (aligned with datasets + contract):

1. `project_evaluation(record)` → `EvaluationRequest` / `EvaluationResponse`.
2. `CapabilityPromptBuilder` renders evaluation system + user (candidate / rubric / expectation).
3. Assistant label = canonical `EvaluationResponse` JSON (`verdict`, `score`, `explanation`, …).

---

## 5. Current dataset assumptions

| Assumption in training | Reality after Evaluation v2 |
|------------------------|-----------------------------|
| One catalog row per file is enough | 189,615 judgment rows |
| Required keys: `evaluation_id`, `metadata` | Required for SFT: judgment fields (`candidate`, `verdict`, …); `record_id` is identity |
| `catalog` present on every training row | Only on BenchmarkCatalog artifact |
| No contract projection possible | `project_evaluation` exists in `aiodoo-datasets` |
| Fixtures = catalog stub | Production SFT ≠ catalog |
| `record_count: 1` in config | Stale |

---

## 6. Compatibility with Evaluation v2 dataset

| Check | Result |
|-------|--------|
| Path name `evaluation_dataset.jsonl` in config | Compatible (filename correct) |
| Required-field validation | **Incompatible** — fails: missing `evaluation_id` |
| Formatter | **Incompatible** — `catalog` absent → trains on `null` catalog; label has no verdict |
| Contract projection in training | **Missing** |
| Prompt protocol (ADR-0003) | **Not used** for Evaluation |
| Tokenizer input shape | Would accept any `TrainingExample`, but content would be wrong |
| Scale / curriculum | Config comments/counts stale; pipeline can scale if formatter fixed |

**Verdict:** Pointing production configs at certified `evaluation_dataset.jsonl` **will not** produce a correct Evaluation LoRA today. With validation on: load fails. With validation off: silent garbage examples.

---

## 7. Compatibility with BenchmarkCatalog

| Check | Result |
|-------|--------|
| Catalog file shape (`evaluation_id` + `catalog` + `metadata`) | **Matches** current formatter + `REQUIRED_FIELDS` |
| `metadata.training_forbidden=true` | Present on catalog; **ignored** by training |
| Guard against training on catalog | **None** (no filename deny-list, no `training_forbidden` check) |
| Intended use | Certification / regression only — **not** SFT |

**Risk:** If an operator (or stale path) points `dataset.yaml` at `evaluation_benchmark_catalog.jsonl`, training **accepts** it and formats “successfully” under the legacy path — exactly the wrong artifact.

Config currently names `evaluation_dataset.jsonl` only (good convention), but safety is **convention-only**.

---

## 8. Every incompatibility discovered

### Blocking

1. **`EvaluationFormatter` is BenchmarkCatalog-shaped** — not judgment R/R.
2. **No `project_evaluation` in `aiodoo_training.contract.adapters`** — Evaluation excluded from `SUPPORTED_CAPABILITIES`.
3. **`REQUIRED_FIELDS` require `evaluation_id`** — rejects certified SFT rows.
4. **Fixtures + unit/contract tests lock in catalog semantics** — green CI does not prove v2 readiness.
5. **No `training_forbidden` / catalog-filename guard** — catalog can be trained accidentally.
6. **Docs/`CONTRACT_ADOPTION.md` still assert “evaluation = BenchmarkCatalog, no projection”** — contradicts datasets v2 + contract.

### Non-blocking (follow-ups during migration)

7. `stable_example_id` does not prefer `record_id` (`EVL-…`) — falls back to content hash.
8. Config `record_count: 1` and README/PRODUCTION_TRAINING stale scale notes.
9. Phase-4 smoke prompts in `evaluation.yaml` still say “evaluation catalog”.
10. Distinguish naming: Phase-4 “evaluation” vs Evaluation capability (docs clarity only).

---

## 9. Recommended fixes

1. Port `project_evaluation` into `aiodoo_training.contract.adapters` (mirror datasets; add to `_PROJECTORS` / `SUPPORTED_CAPABILITIES`).
2. Convert `EvaluationFormatter` to `_ContractFormatter` (or thin wrapper calling `build_training_example`).
3. Update `REQUIRED_FIELDS[EVALUATION]` to judgment SFT keys (e.g. `record_id`/`candidate`/`verdict`/`metadata` — exact set to match datasets schema + training needs).
4. Replace `tests/fixtures/datasets/evaluation.jsonl` with a minimal judgment row; move catalog sample to a non-training fixture if still needed for negative tests.
5. Move Evaluation into contract formatter/contract test cases (3-message examples; `has_contract_projection=True`).
6. Add hard reject: refuse paths containing `benchmark_catalog` and/or records with `metadata.training_forbidden is True`.
7. Prefer `record_id` in `stable_example_id`.
8. Refresh `configs/training/evaluation/*` counts/docs and `CONTRACT_ADOPTION.md` / `PRODUCTION_TRAINING.md` Evaluation rows.
9. Optionally retune Phase-4 smoke prompts to judgment language (not catalog).

---

## 10. Implementation plan

Ordered, no redesign of architecture/contracts/datasets:

| Step | Task | Primary files |
|-----:|------|----------------|
| 1 | Add `project_evaluation` + register capability | `contract/adapters.py` (+ tests) |
| 2 | Switch `EvaluationFormatter` to contract path | `datasets/formatters/formatters.py` |
| 3 | Fix `REQUIRED_FIELDS` + catalog/forbidden guards | `datasets/validation.py` |
| 4 | `record_id` in stable ids | `datasets/mixing.py` |
| 5 | Refresh fixtures | `tests/fixtures/datasets/evaluation.jsonl` |
| 6 | Update formatter + contract tests | `tests/unit/test_formatters.py`, `tests/contract/test_dataset_contracts.py` |
| 7 | Update adoption + production docs/config comments | `CONTRACT_ADOPTION.md`, `PRODUCTION_TRAINING.md`, `configs/training/evaluation/*` |
| 8 | Smoke: validate_ref + format one certified SFT row end-to-end | local / CI |

**Out of scope for the migration:** regenerating datasets; changing `aiodoo-contract` Evaluation schemas; Phase-4 engine redesign; product merge.

---

## Decision (restated)

### B) Evaluation training pipeline is NOT compatible.

Required implementation work: §10 steps 1–8 (blocking items §8.1–§8.6).

Until that lands, **do not train** `aiodoo-evaluation` on the certified v2 corpus. Other certified capability datasets remain usable under their existing contract formatters (separate from this Evaluation audit).
