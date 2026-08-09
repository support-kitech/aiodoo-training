# AT-7.3 — Reasoning Sparse-Skill Data Readiness

**Living.** Follows AT-7.1 (`REASONING_COVERAGE_MIXED`) and AT-7.2
(`AT7_2_TRAINING_PATH_PASS`).

**DATA / SEMANTICS / GENERATION / QUALITY only.**  
Does **not** train Conversation / Approval / Evaluation.  
Does **not** modify Planner, `controlled_batch_2`, Runtime, or Core.

---

## Overall verdict

**REASONING_SPARSE_DATA_PARTIAL**

| Capability | Verdict |
|------------|---------|
| Conversation | **CONVERSATION_CORPUS_READY** |
| Approval | **APPROVAL_CORPUS_READY** |
| Evaluation | **EVALUATION_SEMANTICS_UNRESOLVED** |
| Planner (existing) | Unchanged — AT-7.2 path PASS; full train not this phase |

Blocked capability: **Evaluation** (FP2 generation not authorized).

---

## Corpus version

`fp2-reasoning-sparse-1.0.0`

### Locations

- `aiodoo-training/fixtures/fp2/reasoning_controlled_1/{conversation,approval,evaluation}/`
- `aiodoo-datasets/datasets/fp2/reasoning_controlled_1/{conversation,approval,evaluation}/`

---

## Counts

| Capability | Native | Families | Train | Val | Test |
|------------|-------:|---------:|------:|----:|-----:|
| Conversation | 232 | 58 | 180 | 28 | 24 |
| Approval | 162 | 81 | 134 | 14 | 14 |
| Evaluation | 0 | — | — | — | — |

Family leakage: **0** (both Conversation and Approval).

---

## Record-type / domain distributions

### Conversation

| Metric | Count |
|--------|------:|
| decision_context | 116 |
| loop_decision (clarify) | 116 |
| Odoo | 120 |
| Generic | 112 |

### Approval

| Metric | Count |
|--------|------:|
| loop_decision | 162 |
| approve | 81 |
| reject | 43 |
| modify | 38 |
| Odoo | 86 |
| Generic | 76 |

Engineering WHAT capabilities: **none required** — these tracks teach provider-plane conversational / approval behavior on Continuity families (`decision_context`, `loop_decision`), not Engineering action intents.

---

## Quality scorecard (hard gates)

| Metric | Conversation | Approval |
|--------|-------------:|---------:|
| Exact duplicate groups | 0 | 0 |
| Normalized duplicate groups | 0 | 0 |
| Forbidden HOW | 0 | 0 |
| Taxonomy violations | 0 | 0 |
| Negative contamination | 0 | 0 |
| Legacy contamination | 0 | 0 |
| Family leakage | 0 | 0 |
| Pack validity | true | true |
| provider == dataset_type | true | true |
| Negatives OK | true | true |

---

## Evaluation semantic definition

**System meaning (resolved):** Evaluation is the Reasoning capability that judges
another capability **candidate** against optional **expectation/rubric** and
returns **verdict** (`pass`/`fail`/`inconclusive`), optional **score**, and
**explanation** (`EvaluationRequest` / `EvaluationResponse` in
`aiodoo-contract`).

| Question | Answer |
|----------|--------|
| What does it learn? | Judgment SFT — score/verdict a candidate vs expectation/rubric |
| Inputs | candidate (+ optional expectation, rubric) |
| Outputs | verdict, optional score [0,1], optional explanation |
| Judging another capability? | **Yes** |
| Rubric/verdict/score? | **Yes** |
| Candidate actions / code / plans? | When those appear as the candidate payload |
| FP2 families today | observation / engineering_feedback / decision_context (mapping) |
| Engineering WHAT | None required |

### Why generation stopped

`EVALUATION_SEMANTICS_UNRESOLVED` for **FP2 corpus generation**:

Current mapping families do **not** encode `EvaluationRequest`/`EvaluationResponse`.
Stuffing judgment fields into Continuity/observation families would fabricate a
Training Contract surface. Contract also treats conversation/evaluation on
DecisionContext as unsupported unless proven.

Legacy `evaluation_dataset.jsonl` (~189k) matches meta-judge schema but remains
historical — **do not project** in this phase.

Evidence artifacts: `evaluation/semantics_report.json` (no native JSONL / pack).

---

## Mapping preserved (unchanged)

| Provider | Allowed families |
|----------|------------------|
| Conversation | `decision_context`, `loop_decision` (clarify only in this corpus) |
| Approval | `loop_decision` (approve / reject / modify) |
| Evaluation | no native generation this phase |

Provider filters were **not** widened. Planner corpus was **not** modified.

---

## Negatives (not in training packs)

`REASONING_SPARSE_NEGATIVE_CASES` in `quality/negatives.py`:

- Conversation: forbidden HOW; planner-as-conversation; approval-as-conversation
- Approval: forbidden HOW; planner-as-approval; conversation-as-approval
- Evaluation: meta-judge ambiguity; planner-as-evaluation; approval-as-evaluation
- Positive controls for clarify / approve

---

## controlled_batch_2 immutability

| | Checksum |
|--|----------|
| Before | `728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227` |
| After | `728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227` |

Content hash over native family JSONL files (harness method) unchanged.

### New corpus tree checksums (sorted file digest)

| Tree | SHA-256 |
|------|---------|
| conversation | `488b3a7576071c875c32e277c49562bb9c472904e32b12a1b98fcf6558da9de3` |
| approval | `3e069403348203e3b6aec2ce0f31d2dc622c60146b0bbde6dbfee3134fdfbcb7` |
| evaluation | `579fa3554564d59917c0020a680df52116007bd3ed426b2aad41eb59f40217e0` |

---

## Generators / emit

| Module | Role |
|--------|------|
| `generators/conversation_controlled.py` | Conversation natives |
| `generators/approval_controlled.py` | Approval natives |
| `generators/evaluation_semantics.py` | Semantic definition only |
| `generators/reasoning_sparse_emit.py` | Analyze + emit trees |

Deterministic: same generator → identical record IDs / serialization / splits.

---

## Tests

`tests/unit/test_at73_reasoning_sparse_data.py`

Also re-run TR-2 … TR-7 and impacted Reasoning unit tests (no weaken).

---

## Remaining gaps

1. **Evaluation** FP2 record-type / mapping authorization for judgment surface
2. Conversation / Approval **path smokes** (separate authorized phases)
3. Full Planner train (resource-controlled; not AT-7.3)
4. Optional future: legacy projection phase (explicit only)

---

## Recommended next phase

When authorized:

1. Evaluation mapping / Training Contract surface decision → then Evaluation controlled corpus  
   **or**
2. Conversation path smoke (AT-7.x) using this corpus  
   **or**
3. Approval path smoke using this corpus

Do **not** start AT-8. Do **not** train / merge / certify without explicit auth.

---

## STOP

No training. No adapters. No merge. No certification. No Runtime/Core changes.
No commit/push from this phase without separate authorization.
