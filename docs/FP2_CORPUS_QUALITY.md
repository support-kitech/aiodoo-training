# FP2 Corpus Quality & Training-Pack Readiness (TR-4)

**Living.** Builds on TR-3 fixtures + Canonical Training Contract v1.0.0.  
**Harness:** `aiodoo_training.system_training_contract.quality`  
**Report:** `fixtures/fp2/quality_report_tr4.json` (mirrored under `aiodoo-datasets/datasets/fp2/`)

---

## TR-3 source audit

Verified on disk (both `fixtures/fp2/` and `datasets/fp2/`):

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
| **Total** | **111** |

Legacy production JSONL checksums unchanged.

---

## Quality gates

| Gate | Meaning |
|------|---------|
| PASS | Hard requirement satisfied |
| WARN | Acceptable for controlled TR-5 with documented caveat |
| FAIL | Blocks TR-5 |

Hard gates: fixture inventory, schema, forbidden HOW, taxonomy, serialization,
Work Unit, planning, feedback, continuity, loop decisions, negative corpus.

Soft: capability coverage (≥70% PASS; 40–70% WARN), duplicates, Odoo/generic balance.

---

## Scorecard (current fixtures)

See `quality_report_tr4.json`. Latest run:

- Readiness: **READY_FOR_TR5** (controlled scale)
- Coverage: **19/23 (82.61%)** preferred Engineering IDs
- Uncovered: `artifact.attachment`, `artifact.import`, `diagnostics.collect_logs`, `repository.merge`
- Weak: `repository.history`, `workspace.bind`
- Overrepresented: `validation.run`, `workspace.write`
- Domain: ~36% Odoo / ~64% generic
- Duplicates: 0 groups
- Negatives: 13/13 matched

---

## Forbidden HOW / taxonomy

Model-facing fields scanned; provenance/metadata historical strings allowed.  
Provider packs must not appear as Engineering `capability_id` / plan actions.

---

## Negative / adversarial corpus

`quality_negatives.jsonl` — **quality-only**, `not_for_training=true`.  
Never mix into production training packs.

---

## Training pack formatters

`quality/formatters.py`:

Canonical FP2 record → `TrainingExample` (user/assistant JSON)  
**without** Protocol V1 / `aiodoo_contract` projection.

Packs: `development` / `reasoning` via record-type allowlists (no adapter chaining).

---

## Dataset split strategy

`fp2-split-1.0.0` — scenario-hash buckets 80/10/10.  
Multi-cycle narratives share a family key (no train/test leakage).  
Negatives never train.

---

## TR-5 prerequisites

1. ~~Expand coverage for uncovered preferred IDs~~ — **done in TR-5.2 / batch 1**  
2. Keep hard gates green on generated batches — **batch 1 PASS**  
3. Do not mix negatives or legacy Protocol V1 into FP2 packs  
4. Still no LoRA / Colab / Runtime changes until separately authorized  

TR-5 batch 1: see `docs/FP2_CONTROLLED_BATCH.md`.

TR-7 batch 2 (continuity + domain cleanup): see `docs/FP2_CONTROLLED_BATCH.md`
and `docs/FP2_PACK_EVALUATION.md` — readiness **READY_FOR_TRAINING**.

---

## Run

```bash
PYTHONPATH=. python3 -m aiodoo_training.system_training_contract.quality.cli \
  --corpus fixtures/fp2
```
