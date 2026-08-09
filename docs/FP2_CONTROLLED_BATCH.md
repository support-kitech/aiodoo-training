# FP2 Controlled Corpus Generation (TR-5 / TR-7)

**Living.** Controlled FP2-native batches for System Training Contract v1.0.0.  
**TR-5 batch (immutable evidence):** `fp2-controlled-1.0.0` → `controlled_batch_1/`  
**TR-7 derivative:** `fp2-controlled-2.0.0-tr7` → `controlled_batch_2/`  
**Generators:** `controlled_batch` (TR-5), `tr7_batch` + `tr7_continuity` (TR-7)

---

## Strategy

1. Fill TR-4 coverage gaps with meaningful System scenarios  
2. Coverage-aware expansion (prefer weak/uncovered; dampen overrepresented caps)  
3. Cap first batch at **1,000–1,500** records  
4. Run hard gates + pack evaluation  
5. Deterministic 80/10/10 splits (`fp2-split-1.0.0`)  
6. Emit Development / Reasoning packs  
7. **TR-7:** expand Reasoning continuity; clean Odoo/generic labels on a
   versioned derivative (never silently overwrite TR-5)

---

## Batch 1 results (TR-5 — preserved)

| Metric | Value |
|--------|------:|
| Decision | **PASS** |
| Native records | 1200 |
| Preferred Engineering coverage | **23/23 (100%)** |
| engineering_state / decision_context / loop_decision | 15 / 15 / 15 |
| Development / Reasoning packs | 942 / 954 |
| Odoo / generic (raw labels) | 468 / 732 |

---

## Batch 2 results (TR-7)

| Metric | Value |
|--------|------:|
| Decision (TR-6 harness re-eval) | **READY_FOR_TRAINING** |
| Native records | 1386 |
| Continuity added | 186 (62 scenario families) |
| engineering_state / decision_context / loop_decision | **77 / 77 / 77** |
| Ambiguous domain labels | **0** |
| Domain actions | clear_odoo 303, set_odoo 1, keep 896 |
| Quarantined (non-training) | 0 |
| Development / Reasoning packs | 1004 / 1078 |
| Train / val / test | 1115 / 133 / 138 |
| Odoo / generic (semantic) | 226 / 1160 |

### Continuity methodology

Multi-cycle narratives covering the required families (success→COMPLETE,
op-success/incomplete→REPLAN, failure→REPLAN/ESCALATE, repair continua,
prior COMPLETE vs current failure isolation, empty/partial/conflicting
evidence, blockers, multi-work-unit, bounded history, etc.). Invariants:
history ≠ current; empty evidence ≠ COMPLETE; no automatic
failure→repair→validation→complete pipeline; no Memory/RAG; no forbidden HOW.

### Odoo classification methodology

Semantic cues (`models/partner`, `action_confirm`, `__manifest__`, `addons/`,
`res.partner`, ORM markers, …). Provenance mentioning Odoo alone does **not**
force specialization. Unjustified `odoo` labels cleared; cue-bearing unlabeled
records set to `odoo`. Remaining unprovable cases would be quarantined to
`ambiguous_quarantine.jsonl` (none required in TR-7).

---

## Contents

| Artifact | Role |
|----------|------|
| `controlled_batch_1/` | TR-5 evidence (do not overwrite) |
| `controlled_batch_2/*.jsonl` | TR-7 corrected + expanded natives |
| `controlled_batch_2/tr7_domain_audit.jsonl` | Domain correction traceability |
| `controlled_batch_2/ambiguous_quarantine.jsonl` | Explicit non-training quarantine |
| `controlled_batch_2/pack_*.jsonl` | Regenerated Development/Reasoning packs |
| `controlled_batch_2/splits.jsonl` | Regenerated `fp2-split-1.0.0` splits |
| `controlled_batch_2/quality_report_tr7.json` | Re-evaluation report |
| `quality_negatives.jsonl` (parent) | Negatives — never in packs/splits |

---

## Regenerate TR-7 derivative

```bash
cd ../aiodoo-training
PYTHONPATH=. python3 -m aiodoo_training.system_training_contract.generators.tr7_batch
```

---

## Out of scope

LoRA, foundation downloads, adapter packaging, legacy JSONL rewrite, Runtime edits.
