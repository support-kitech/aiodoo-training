# AT-7.4 — Evaluation FP2 Training Contract & Mapping Decision

**Living.** Follows AT-7.3 (`REASONING_SPARSE_DATA_PARTIAL`).

**SEMANTICS / CONTRACT / MAPPING DESIGN only.**  
Does **not** generate Evaluation corpus. Does **not** train.  
Does **not** modify `controlled_batch_2`, Conversation, Approval, or Planner.

---

## Verdict

**EVALUATION_MAPPING_READY**

System EvaluationRequest/Response was already complete. Training Contract now
includes the smallest FP2 family that encodes that surface:
`evaluation_judgment`. Provider mapping restricts Evaluation to that family only.

Not `EVALUATION_CORPUS_READY` — no corpus generated.

---

## Source evidence

| Source | Finding |
|--------|---------|
| `aiodoo_contract.schemas.evaluation.EvaluationRequest` | `candidate` required; `expectation?`, `rubric?` (rubric is `str`) |
| `aiodoo_contract.schemas.evaluation.EvaluationResponse` | `verdict` required; `score?` ∈ [0,1]; `explanation?` |
| `EvaluationVerdict` | `pass` / `fail` / `inconclusive` |
| Request docstring | Candidate/expectation are untyped dicts — generic over any capability dump |
| `CapabilityPromptBuilder._extract_evaluation` | Prompt = Candidate + optional Rubric/Expectation sections |
| `REASONING_CAPABILITIES` | Includes `evaluation` |
| `docs/capability_model.md` | Judgment SFT; not BenchmarkCatalog |
| Core Runtime | Product-plane `evaluation` invoke exists; distinct from Engineering objective evaluation / validation.run |
| Legacy `evaluation_dataset.jsonl` | ~189k meta-judge rows matching request/response shape |

**Doc vs source:** AT-7.3 / Training Contract previously allowed Evaluation on
`observation` / `engineering_feedback` / `decision_context`. Source System
schemas do **not** map to those families. Followed source → removed those
mappings.

---

## What the Evaluation adapter teaches

The Evaluation adapter teaches the model to emit **structured judgments**:

given a candidate (+ optional expectation/rubric) → produce verdict / score /
explanation.

It does **not** teach:

| Not Evaluation | Why |
|----------------|-----|
| Planner | Produces plans/steps, not verdicts |
| Conversation | Clarifies with users |
| Approval | approve/reject/modify loop decisions |
| Execution / Repair / Coding | Produce or apply Engineering WHAT |
| Context | Locate/retrieve |
| Runtime validation / oracles | System infrastructure executing checks |

Models produce judgment structure. Agents/Runtime may invoke Evaluation as a
capability; Training must not become Runtime.

---

## Semantic definition (required answers)

| # | Topic | Decision |
|---|-------|----------|
| 1 | Input | `candidate` (+ optional `expectation`, `rubric`) |
| 2 | Output | `verdict` (+ optional `score`, `explanation`) |
| 3 | Candidate | Generic `dict` — typically another capability's dumped request/response |
| 4 | Expectation | Optional `dict` reference outcome |
| 5 | Rubric | Optional `str` criteria text (System schema) |
| 6 | Verdict | `pass` \| `fail` \| `inconclusive` |
| 7 | Score | Optional float in **[0.0, 1.0]** |
| 8 | Explanation | Optional string rationale |
| 9 | Required | `candidate`, `verdict`; provider must be `evaluation` |
| 10 | Inconclusive | First-class verdict when judgment cannot be decided |
| 11 | Binary vs scalar | **Both** — verdict required; score optional |
| 12 | Multiple criteria | Via rubric text / expectation structure — not a separate multi-criteria schema |
| 13 | Evidence in request | **No** System field — use metadata if needed |
| 14 | Evidence in response | **No** — explanation only |
| 15–19 | Plans/code/tools/conversations/other caps | **Yes**, as generic candidate payloads |
| 20 | Representation | **One** family `evaluation_judgment` — not per-judged-capability families |

---

## Existing FP2 family analysis

| Family | Can represent Evaluation? | Why |
|--------|---------------------------|-----|
| capability_intent | **No** | Engineering action intent |
| observation | **No** | Execution observation envelope |
| execution_work_unit | **No** | Work Unit WHAT |
| planning_decision | **No** | Planner steps/COMPLETE |
| decision_context | **No** | Continuity bridge; stuffing judgment fabricates surface |
| loop_decision | **No** | Loop/approval/clarify decisions |
| engineering_state | **No** | Cycle state |
| engineering_feedback | **No** | Engineering objective feedback ≠ capability Evaluation |

---

## New record type decision

**Required:** `evaluation_judgment`

| Field | Value |
|-------|-------|
| Name | `evaluation_judgment` |
| Purpose | Mirror EvaluationRequest → EvaluationResponse |
| system_contract | `capability.evaluation` |
| Required | `candidate`, `verdict`, `provider_capability=evaluation` |
| Optional | `expectation`, `rubric`, `score`, `explanation` |
| Provider | `evaluation` only |
| dataset_type | `evaluation` (from provider) |
| Split / scenario_family | Via `metadata.scenario_family` (same as other FP2) |
| Engineering WHAT | **none** |

`NATIVE_FAMILIES` (batch_2 inventory) **unchanged** — additive Training Contract
type; not part of immutable batch_2.

---

## Provider mapping decision

**Allowed:** `evaluation_judgment` → `{evaluation}`

**Rejected for Evaluation:** all other families (including former
observation/feedback/decision_context Evaluation labels).

batch_2 still contains 3 historical Evaluation placeholder labels on old
families — files immutable; they are **not** the authorized Evaluation
surface going forward.

---

## Engineering WHAT

**None.** Evaluation is provider-plane judgment, not an Engineering action ID.

---

## Negative controls

Extended `REASONING_SPARSE_NEGATIVE_CASES`:

- missing candidate, invalid verdict, score out of range, wrong provider, forbidden HOW
- planner/approval/conversation/execution/observation-as-evaluation (policy)
- meta-judge stuffed decision_context
- positive control `pos_eval_judgment_control`

Must not enter training packs.

---

## Legacy Evaluation analysis

| Item | Value |
|------|-------|
| Path | `aiodoo-datasets/datasets/evaluation_dataset.jsonl` |
| Approx size | ~189,615 rows |
| Schema | candidate, expectation, rubric, verdict, score, explanation + meta |
| Nature | Meta-judge over other capability candidates |
| Maps to System? | Shape-aligned with EvaluationRequest/Response |
| **LEGACY_PROJECTION** | **NOT PERFORMED** |

Recommend future phase: **Evaluation Legacy → FP2 Projection** (explicit only).

---

## Training Contract changes

Additive within `training_contract_version=1.0.0`:

- `EvaluationJudgmentRecord` + `RECORD_TYPES`
- mapping allow/reject sets
- formatter default `evaluation_judgment` → `DatasetType.EVALUATION`
- contract doc table + projection rule update
- lazy import fix in `formatters.format_fp2_pack` (circular import safety)

Unrelated providers unchanged.

---

## System Contract changes

**None.** `aiodoo-contract` already defines EvaluationRequest/Response.

---

## Immutability

| Item | Result |
|------|--------|
| batch_2 before | `728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227` |
| batch_2 after | identical |
| Conversation 232 | preserved |
| Approval 162 | preserved |
| Planner AT-7.2 | untouched |
| Evaluation natives | still **0** |

---

## Tests

`tests/unit/test_at74_evaluation_contract.py` + TR-2…TR-7 + impacted Reasoning tests.

---

## Remaining gaps

1. Controlled Evaluation corpus (AT-7.5)
2. Optional legacy projection phase
3. Conversation / Approval path smokes (independent)

---

## Exact next phase

**AT-7.5 — Controlled Evaluation FP2 Corpus Generation & Readiness Audit**

Generate `evaluation_judgment` natives only; scenario families; splits; quality;
no legacy mix; no Planner/Conversation/Approval contamination.

---

## STOP

No training. No Evaluation corpus. No adapter. No merge. No certification.
No legacy projection. No Runtime/Core changes. No commit/push without auth.
