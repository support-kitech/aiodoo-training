# FP2-Native Corpora (TR-3)

**Living.** Builds on Canonical Training Contract v1.0.0.  
**Package:** `aiodoo_training.system_training_contract.generators`  
**Fixtures:** `fixtures/fp2/` (training) and `aiodoo-datasets/datasets/fp2/`

---

## Generator architecture

Independent generators per System WHAT surface (not one giant generator):

| Generator | Record type | Primary adapter packs |
|-----------|-------------|------------------------|
| `capability_intent` | `capability_intent` | planner, execution, repair, coding |
| `work_unit` | `execution_work_unit` | execution, planner, repair |
| `planning` | `planning_decision` | planner |
| `observation` | `observation` | execution, repair, evaluation |
| `feedback` | `engineering_feedback` | planner, execution, evaluation |
| `state` | `engineering_state` | planner, execution |
| `decision_context` | `decision_context` | planner, conversation, evaluation |
| `loop_decision` | `loop_decision` | planner, approval, conversation |

Emit orchestration: `generators/emit.py` → validated JSONL + `manifest.json`.

No adapter chaining. Development and Reasoning remain independent.

---

## Fixture counts (current)

| Family | Count |
|--------|------:|
| capability_intent | 16 |
| execution_work_unit | 12 |
| planning_decision | 12 |
| observation | 16 |
| engineering_feedback | 12 |
| engineering_state | 12 |
| decision_context | 12 |
| loop_decision | 12 |
| projection fixtures | 7 |
| **Total records** | **111** |

---

## Historical projection

Selective fixtures only (`projection_fixtures.jsonl`). Production legacy JSONL untouched.

Statuses exercised: projected / partially_projected / unsupported / rejected.

---

## Odoo specialization

`domain_specialization=odoo` on a subset of fixtures; generic examples coexist under the same contract.

---

## Out of scope (TR-3)

- Mass regeneration of legacy corpora
- LoRA / Colab / model download
- Adapter packaging / certification redesign
- Runtime changes

## TR-4 follow-on

Quality harness + readiness: **COMPLETE** — see `docs/FP2_CORPUS_QUALITY.md`.  

## TR-5

Controlled batch 1: **COMPLETE / PASS** — see `docs/FP2_CONTROLLED_BATCH.md`  
(`datasets/fp2/controlled_batch_1/`, 1200 records, 23/23 coverage).

## TR-6

Pack evaluation on batch_1: **COMPLETE** — **READY_WITH_REQUIRED_DATA_FIXES**  
See `docs/FP2_PACK_EVALUATION.md`.

## TR-7

P1 fixes + re-evaluation: **COMPLETE** — **READY_FOR_TRAINING** on
`controlled_batch_2` (`fp2-controlled-2.0.0-tr7`). Continuity 77/77/77;
ambiguous domain labels 0. Still **no** adapter training until separately
authorized.
