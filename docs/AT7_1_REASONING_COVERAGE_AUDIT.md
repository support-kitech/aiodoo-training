# AT-7.1 — Reasoning Skill Corpus Coverage Audit (READ-ONLY)

**Living.** Follows AT-6.3 (`AT6_3_TRAINING_PATH_PASS`).  
**AUDIT ONLY.** Does **not** train Planner / Conversation / Approval / Evaluation.  
Does **not** modify corpora, generators, mapping, configs, or Runtime.

---

## 1. Overall verdict

**REASONING_COVERAGE_MIXED**

| Provider | Classification | Native N |
|----------|----------------|---------:|
| **planner** | **COVERAGE_CONFIRMED** | **657** |
| **conversation** | **DATA_PHASE_REQUIRED** | **8** |
| **approval** | **DATA_PHASE_REQUIRED** | **9** |
| **evaluation** | **DATA_PHASE_REQUIRED** | **3** |

Planner is taxonomy-correct and large enough for a later Reasoning path smoke.
Conversation / Approval / Evaluation are **not** selection bugs: labels are
correct and A≡B in packs, but populations are fixture-scale (plus thin TR-7
continuity for conversation/approval) — insufficient for meaningful adapter
training without an authorized data/generator phase.

---

## 2. Corpus version / checksum

| Item | Value |
|------|-------|
| Path | `aiodoo-datasets/datasets/fp2/controlled_batch_2/` |
| Version | `fp2-controlled-2.0.0-tr7` |
| Checksum before | `728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227` |
| Checksum after | **identical** (corpus untouched) |
| Native population | **1386** |

---

## 3–5. Provider / pack distributions

### Native `provider_capability`

| Provider | Count |
|----------|------:|
| planner | **657** |
| execution | 641 |
| repair | 42 |
| coding | 26 |
| approval | **9** |
| conversation | **8** |
| evaluation | **3** |
| context | 0 |

### Development pack (`pack_development.jsonl`, 1004)

| dataset_type / provider | Count |
|-------------------------|------:|
| execution | 640 |
| planner | **294** |
| repair | 42 |
| coding | 26 |
| evaluation | **2** |
| conversation | **0** |
| approval | **0** |

### Reasoning pack (`pack_reasoning.jsonl`, 1078)

| dataset_type / provider | Count |
|-------------------------|------:|
| planner | **580** |
| execution | 425 |
| repair | 29 |
| coding | 24 |
| approval | **9** |
| conversation | **8** |
| evaluation | **3** |

**Why planner appears in both packs:** shared record types
(`engineering_feedback` ∈ both planes; `engineering_state` Development-only but
planner-labeled). Formatter includes a native when its `record_type` is allowed
for the pack — not a provider-vs-dataset mismatch. Within each pack,
`provider_capability ≡ dataset_type` for all four Reasoning providers (**0**
mismatches).

---

## 6. Planner coverage analysis — **COVERAGE_CONFIRMED**

| Metric | Value |
|--------|------:|
| Native A | **657** |
| Reasoning-pack B | **580** |
| Development-pack B | **294** |
| A≡B within packs | **Yes** |
| Train / val / test | **524 / 62 / 71** |

**Record types (native):**

| record_type | Count |
|-------------|------:|
| planning_decision | 228 |
| engineering_feedback | 217 |
| engineering_state | 77 |
| decision_context | 71 |
| loop_decision | 64 |
| capability_intent | **0** |

**Where generated:** `planning.py`, `feedback.py`, `state.py`,
`decision_context.py`, `loop_decision.py`, `controlled_batch.py`,
`tr7_continuity.py` — overwhelmingly `provider_capability="planner"`.

**Notes:**

- Mapping allows planner on many families including `capability_intent`, but
  TR-3 fixtures and TR-7 natives do **not** emit planner `capability_intent`
  (CI fixtures are execution/coding/repair). Not a filter under-count of
  existing planner labels.
- Continuity variants produce **22** exact-fingerprint duplicate groups
  (alt_surface / alt_repo narratives). Soft quality P2 for full train; does
  **not** invalidate COVERAGE_CONFIRMED or path smoke.
- Domain: Odoo 114 / generic 543.

**Training recommendation:** Path smoke **appropriate** when authorized (AT-3 /
AT-5.1 style). Corpus expansion **not required** before smoke. Full train
possible later after smoke; not started here.

---

## 7. Conversation coverage analysis — **DATA_PHASE_REQUIRED**

| Metric | Value |
|--------|------:|
| Native A | **8** |
| Reasoning-pack B | **8** |
| Development-pack B | **0** |
| A≡B | **Yes** |
| Train / val / test | **8 / 0 / 0** |

**Record types:** `decision_context` 4, `loop_decision` 4 (`clarify`).

**Where generated:**

- Fixture `decision_context.py`: 1 conversation spec
- Fixture `loop_decision.py`: 1 clarify → conversation
- TR-7 continuity: expands empty-evidence / clarify narratives → conversation

**Mapping:** conversation allowed only on `decision_context` and
`loop_decision` — correct, not a mapping omission.

**Why N=8:** **Controlled-data / generator policy** (few conversation specs),
not pack-selection failure. Taxonomy declares conversation as a Reasoning
provider; FP2 never built a Conversation-scale corpus (unlike Planner).

**Split gap:** **no validation or test** Conversation records — blocks a
standard smoke that needs val.

**Legacy:** `conversation_dataset.jsonl` (**29016**) — Protocol-style turns;
no FP2 fields. Relevant for a future explicit projection phase only.

**Not** SELECTION_BUG. Not pure INTENTIONALLY_SPARSE for training readiness
(taxonomy expects an independent skill; N=8 + train-only is not training-ready).

**Recommendation:** **Data/generator phase first.** Do **not** path-smoke or
full-train on 8 train-only records. Optional later: legacy → FP2 projection
(separate authorization).

---

## 8. Approval coverage analysis — **DATA_PHASE_REQUIRED**

| Metric | Value |
|--------|------:|
| Native A | **9** |
| Reasoning-pack B | **9** |
| Development-pack B | **0** |
| A≡B | **Yes** |
| Train / val / test | **8 / 0 / 1** |

**Record types:** `loop_decision` only (approve 3 / reject 3 / modify 3).

**Where generated:**

- Fixture `loop_decision.py`: 3 approval kinds
- TR-7 continuity: approve/reject/modify variants
- `decision_context.py` intentionally remaps `provider="approval"` →
  **`provider_capability="planner"`** (approval decisions use `loop_decision`)

**Mapping:** approval only on `loop_decision` — matches contract
(“approve/reject/modify → loop_decision”).

**Why N=9:** Generator/fixture sparsity + continuity thin expansion; not a
relabel opportunity on unrelated families.

**Legacy:** `approval_dataset.jsonl` (**17094**) — review/decision/findings
schema; not FP2-native.

**Recommendation:** **Data phase first.** Path smoke blocked until population
and splits are adequate. Do not convert legacy silently.

---

## 9. Evaluation coverage analysis — **DATA_PHASE_REQUIRED**

| Metric | Value |
|--------|------:|
| Native A | **3** |
| Reasoning-pack B | **3** |
| Development-pack B | **2** |
| A≡B within packs | **Yes** |
| Train / val / test | **2 / 0 / 1** |

**Record types:** `observation` 1, `engineering_feedback` 1, `decision_context` 1.

**Where generated:** single evaluation-labeled specs in `observation.py`,
`feedback.py`, `decision_context.py`.

**Mapping:** evaluation allowed on `observation`, `engineering_feedback`,
`decision_context`.

**Why N=3:** Extreme fixture under-emission relative to declared Reasoning
taxonomy. Not a pack filter bug (all 3 appear in Reasoning pack).

**Legacy caution:** `evaluation_dataset.jsonl` (**189615**) is a **meta-judge**
corpus (`capability_under_test`, rubric, verdict/score over other skills’
candidates). That is **not** proven equivalent to the Reasoning
`provider_capability=evaluation` adapter skill. Do **not** auto-project.

**Recommendation:** **Data phase required** after clarifying Evaluation
provider semantics vs meta-evaluation datasets. Path smoke / full train
**blocked**. Possibly defer Evaluation adapter until System evidence proves
the skill surface.

---

## 10. A / B / C comparison (provider vs dataset_type)

Per pack (metadata.provider_capability = A, dataset_type = B, intersection = C):

| Provider | Pack | \|A\| | \|B\| | \|C\| | A≡B≡C |
|----------|------|------:|------:|------:|-------|
| planner | development | 294 | 294 | 294 | **Yes** |
| planner | reasoning | 580 | 580 | 580 | **Yes** |
| conversation | development | 0 | 0 | 0 | Yes (empty) |
| conversation | reasoning | 8 | 8 | 8 | **Yes** |
| approval | development | 0 | 0 | 0 | Yes (empty) |
| approval | reasoning | 9 | 9 | 9 | **Yes** |
| evaluation | development | 2 | 2 | 2 | **Yes** |
| evaluation | reasoning | 3 | 3 | 3 | **Yes** |

**Conclusion:** Future filters on
`metadata.provider_capability==X AND dataset_type==X` will **not** under-select
relative to labeled natives inside a given pack. Sparse skills fail on **N**,
not on A≠B.

Native vs pack-sum for planner (657 vs 874) is **dual-pack inclusion** of
shared families, not identity mismatch.

---

## 11. Record-family mapping (source)

From `generators/mapping.py` `_RECORD_TO_PROVIDERS`:

| Record type | Reasoning providers allowed |
|-------------|----------------------------|
| capability_intent | planner (+ Dev) |
| execution_work_unit | planner (+ Dev) |
| planning_decision | **planner only** |
| observation | evaluation (+ Dev) |
| engineering_feedback | planner, evaluation (+ Dev) |
| engineering_state | planner (+ Dev) |
| decision_context | planner, conversation, evaluation |
| loop_decision | planner, approval, conversation |

Rejected for conversation/approval outside those families — by design.

---

## 12. Generator sources (where labels are actually emitted)

| Provider | Emitters |
|----------|----------|
| planner | `planning.py`, `feedback.py`, `state.py`, `decision_context.py`, `loop_decision.py`, `controlled_batch.py`, `tr7_continuity.py` |
| conversation | `decision_context.py` (few), `loop_decision.py` (clarify), `tr7_continuity.py` (clarify families) |
| approval | `loop_decision.py` (approve/reject/modify), `tr7_continuity.py` |
| evaluation | `observation.py` (1), `feedback.py` (1), `decision_context.py` (1) |

TR-3 fixture emission snapshot (not batch_2): conversation≈2, approval≈3,
evaluation≈3 — confirms fixture-scale policy carried into TR-7.

---

## 13. Split analysis

| Provider | train | validation | test |
|----------|------:|-----------:|-----:|
| planner | 524 | 62 | 71 |
| conversation | 8 | **0** | **0** |
| approval | 8 | **0** | 1 |
| evaluation | 2 | **0** | 1 |

Scenario-family leakage (multi-split families): planner 1 soft family group;
conversation/approval/evaluation **0**.

---

## 14. Quality checks (read-only)

| Check | Planner | Conversation | Approval | Evaluation |
|-------|---------|--------------|----------|------------|
| STC version | all 1.0.0 | 1.0.0 | 1.0.0 | 1.0.0 |
| Negatives in natives | 0 | 0 | 0 | 0 |
| Legacy meta flags | 0 | 0 | 0 | 0 |
| Forbidden HOW (model-facing rough) | 0 | 0 | 0 | 0 |
| Exact fingerprint dup groups | 22 (continuity variants) | 1 | 3 | 0 |
| Provider≠dataset_type in packs | 0 | 0 | 0 | 0 |

---

## 15. Legacy dataset inventory (no projection)

| File | Count | FP2 fields? | Relevance |
|------|------:|-------------|-----------|
| `planner_v1_0.jsonl` | 5695 | No | Protocol planner plans — future projection candidate |
| `planner_eval_corpus.jsonl` | 50 | No | Eval harness |
| `conversation_dataset.jsonl` | 29016 | No | Turn dialogue — Conversation projection candidate |
| `conversation_eval_corpus.jsonl` | 50 | No | Eval harness |
| `approval_dataset.jsonl` | 17094 | No | Review decisions — Approval projection candidate |
| `approval_eval_corpus.jsonl` | 50 | No | Eval harness |
| `evaluation_dataset.jsonl` | 189615 | No | **Meta-judge** of other capabilities — **not** proven = provider evaluation |
| `evaluation_eval_corpus.jsonl` | 50 | No | Eval harness |
| `evaluation_benchmark_catalog.jsonl` | 1 | No | Catalog |

Do **not** mix into FP2 without an authorized projection phase.

---

## 16. Intentional vs gap classification

| Provider | Class | Rationale |
|----------|-------|-----------|
| planner | **COVERAGE_CONFIRMED** | Large labeled set; mapping+generators emit correctly; splits healthy; A≡B |
| conversation | **DATA_PHASE_REQUIRED** | Taxonomy expects skill; N=8 train-only; generator sparsity + no val/test |
| approval | **DATA_PHASE_REQUIRED** | Taxonomy expects skill; N=9; loop_decision-only by policy; no val |
| evaluation | **DATA_PHASE_REQUIRED** | Taxonomy lists skill; N=3; semantics vs meta-eval legacy unresolved |

None classified **SELECTION_BUG** (filters would not under-count labels).
None classified pure **COVERAGE_GAP** in the AT-6.1 sense (labels exist; sparse
by generator policy). Conversation/Approval/Evaluation are **not** falsely
“complete” merely because A≡B.

---

## 17. Training recommendation (do not start)

| Provider | Path smoke now? | Corpus expansion first? | Full training? | Blocked? |
|----------|-----------------|-------------------------|----------------|----------|
| **planner** | **Yes** (when authorized) | No (optional later for CI planner intents / dedupe) | After smoke | No |
| **conversation** | No | **Yes** (FP2-native; optional legacy projection separate) | No | **Yes** until data phase |
| **approval** | No | **Yes** | No | **Yes** until data phase |
| **evaluation** | No | **Yes** (after semantic definition) | No | **Yes** until data + semantics |

---

## 18. Files changed

**Audit documentation only:**

- `docs/AT7_1_REASONING_COVERAGE_AUDIT.md` (this file)
- status pointers in `docs/STATUS.md` / `docs/TRAINING_CONTRACT_TARGET.md`

**No** corpus, generator, mapping, config, or Runtime changes.

---

## STOP

Do **not** train Planner / Conversation / Approval / Evaluation.  
Do **not** expand datasets or project legacy.  
Do **not** create adapters, merge, certify, or start AT-8.

Await explicit authorization for the next phase (recommended: **Planner path
smoke**, or a Reasoning sparse-skill data phase for conversation/approval/evaluation).
