# AT-3.1 — Coding Corpus Coverage Audit (READ-ONLY)

**Living.** Follows AT-3 (`AT3_TRAINING_PATH_PASS`).  
**Does not** train, modify corpora, or change selection code.

---

## Verdict

**CODING_COVERAGE_CONFIRMED**

The AT-3 filter selected **all** FP2 records currently labeled
`provider_capability=coding`. The count **26** is the complete coding-labeled
population in `controlled_batch_2`, not an under-count caused by a defective
intersection of `provider_capability` and `dataset_type`.

| Question | Answer |
|----------|--------|
| Is AT-3 selection dropping labeled coding records? | **No** |
| Are `provider_capability=coding` and `dataset_type=coding` different sets? | **No** — identical (26) |
| Why only 26 of 1004 Development examples? | **Generator assignment** — most Development natives are labeled `execution` / `repair` / `planner` |
| Should AT-3 have trained on execution-labeled workspace records as coding? | **No** under current source semantics (provider plane ≠ Engineering `capability_id`) |

---

## Checksums

| When | SHA-256 of native family JSONLs |
|------|----------------------------------|
| Before | `728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227` |
| After | **unchanged** (read-only) |

Matches `manifest.json` / AT-3 provenance.

---

## Corpus inventory

| Metric | Count |
|--------|------:|
| Total native records | 1386 |
| Development pack | 1004 |
| Reasoning pack | 1078 |
| Splits (`fp2-split-1.0.0`) | train 1115 / val 133 / test 138 |

### Native `provider_capability`

| Provider | N |
|----------|--:|
| planner | 657 |
| execution | 641 |
| repair | 42 |
| **coding** | **26** |
| approval | 9 |
| conversation | 8 |
| evaluation | 3 |

### Native `record_type`

| Type | N |
|------|--:|
| observation | 237 |
| capability_intent | 235 |
| execution_work_unit | 231 |
| planning_decision | 228 |
| engineering_feedback | 224 |
| engineering_state | 77 |
| decision_context | 77 |
| loop_decision | 77 |

All natives: `training_contract_version=1.0.0`, no `legacy=true` flags.

---

## Coding set comparison (critical)

| Selection | Count | Train / Val / Test |
|-----------|------:|--------------------|
| A. `provider_capability=coding` | **26** | 22 / 1 / 3 |
| B. `dataset_type=coding` | **26** | 22 / 1 / 3 |
| C. A ∧ B (AT-3 filter) | **26** | 22 / 1 / 3 |

`A ≡ B ≡ C` (same `record_id` sets).  
AT-3 derived `artifacts/at3_coding/data/*` matches C exactly.

**Why A≡B:** `format_fp2_record` → `_dataset_for_record` sets
`dataset_type = DatasetType(provider_capability)` when provider is set
(`quality/formatters.py`). `dataset_type` is **not** an independent
discriminator; it mirrors provider.

---

## Coding composition (the 26)

| `record_type` | N | Mapping allowlist includes `coding`? |
|---------------|--:|--------------------------------------|
| capability_intent | 23 | **Yes** |
| execution_work_unit | 2 | **No** (`execution`,`planner`,`repair` only) |
| observation | 1 | **No** (`execution`,`repair`,`evaluation` only) |

Engineering `capability_id` on coding intents: `workspace.search` 9,
`workspace.read` 8, `workspace.bind` 6, `workspace.write` 1,
`workspace.navigate` 1.

Domain: odoo 19 / unset 7.

Sources: fixture generators (`capability_intent`, `execution_work_unit`,
`observation`) + controlled-batch intents. **batch_1 coding natives also = 26**
(TR-7 did not expand coding labels).

---

## Exact reason AT-3 got 26

Path:

```
FP2 native (provider_capability set by generators)
  → format_fp2_pack(pack="development")  # by DEVELOPMENT_RECORD_TYPES
  → TrainingExample.dataset_type := provider_capability
  → pack_development.jsonl (1004)
  → AT-3 filter: provider_capability==coding AND dataset_type==coding
  → 26 examples → split join via record_id
```

Excluded from Development pack (978):

| Reason | N |
|--------|--:|
| `provider_capability=execution` | 640 |
| `provider_capability=planner` | 294 |
| `provider_capability=repair` | 42 |
| `provider_capability=evaluation` | 2 |

These are **legitimately non-coding under current labels** (independent skill
adapters filter on `provider_capability`). Shared record *types* may appear in
the Development pack with non-Development providers (planner/evaluation) because
`format_fp2_pack` filters primarily by `record_type`, not by Development provider
set — that is pack composition, not a coding-selection bug.

---

## Shared-record / generator semantics (source)

`generators/mapping.py` documents shared record types across providers.
Each **instance** still carries exactly one `provider_capability`.

Controlled-batch rule (`controlled_batch.py`):

- `execution.repair` → `repair`
- `workspace.*` **and** `domain=odoo` → `coding` (**intent only**)
- else → `execution`
- Sibling WU/observation for the same scenario are forced to `execution`/`repair`,
  **not** duplicated as coding

Fixture `capability_intent._SPECS` explicitly marks only four intents as coding;
`fp2-ci-002` is `workspace.bind` + odoo but labeled **execution** by fixture
(intentional divergence from the controlled-batch rule).

Therefore many `workspace.*` Engineering IDs exist under `execution` — that is
**provider specialization policy**, not AT-3 dropping coding metadata.

---

## Quality / safety (audit)

| Check | Result |
|-------|--------|
| Negatives in coding-26 | none |
| Legacy Protocol V1 in coding-26 | none (`fp2_native` pack) |
| Narrative family leakage across splits | none |
| Forbidden HOW in selection | none observed in audit pass |
| `controlled_batch_2` modified | **No** |

---

## Residual notes (not selection defects)

| ID | Note | Severity |
|----|------|----------|
| N1 | Coding labeled population is small (26) for *quality* learning | P2 data volume |
| N2 | 3 coding EWU/observation rows exist though mapping allowlist omits coding for those types | P3 mapping/generator drift (rows are **included**, not excluded) |
| N3 | Fixture `fp2-ci-002` (workspace.bind+odoo→execution) vs controlled-batch coding rule | P3 policy inconsistency |
| N4 | No `context` provider labels in this corpus | P3 coverage gap (out of scope for coding) |

Expanding coding requires a **controlled generator/policy change** (future data
phase) — not an AT-3 filter patch, and not silent record duplication.

---

## Recommendation

1. Treat AT-3 coding selection as **correct** for current FP2 labels.  
2. Do **not** widen AT-3 filter to `execution` workspace records without an
   explicit taxonomy decision.  
3. Before large coding retrain: authorize a data/generator phase if product
   wants a larger coding specialization set.  
4. AT-4 repair may proceed on the same provider-filter pattern; expect a
   similarly small but coherent labeled set (42 repair natives).

STOP — no training, no corpus edits, no AT-4 in this phase.
