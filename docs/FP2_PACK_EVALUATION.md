# FP2 Corpus Evaluation & Training-Pack Readiness (TR-6 / TR-7)

**Living.** Evaluates FP2 controlled batches for adapter-training suitability.  
**Does not train.**  
**Harness:** `aiodoo_training.system_training_contract.evaluation`  
**TR-5 evidence (immutable):** `controlled_batch_1/`  
**TR-7 derivative:** `controlled_batch_2/` (`fp2-controlled-2.0.0-tr7`)

---

## Current decision (TR-7)

**READY_FOR_TRAINING**

Evaluated on `controlled_batch_2`. Both TR-6 P1 blockers are resolved.
Hard gates remain PASS. No adapter training has been started.

---

## TR-6 decision (historical, batch_1)

**READY_WITH_REQUIRED_DATA_FIXES**

Structural gates PASS; P1: continuity volume (15/15/15) + ambiguous Odoo/generic
labels. See archive section below.

---

## TR-7 inventory (source-verified)

| Item | Before (batch_1) | After (batch_2) |
|------|-----------------:|----------------:|
| Native records | 1200 | 1386 |
| engineering_state | 15 | 77 |
| decision_context | 15 | 77 |
| loop_decision | 15 | 77 |
| Development pack | 942 | 1004 |
| Reasoning pack | 954 | 1078 |
| Ambiguous domain labels | 203 (TR-6 narrow) / 304 (TR-7 semantic) | **0** |
| Odoo / generic (semantic) | — | 226 / 1160 (16.3% Odoo) |
| Train / val / test | 956 / 124 / 120 | 1115 / 133 / 138 |
| SHA-256 match | yes | yes |

---

## Hard gates (batch_2)

All PASS: inventory band, checksum, schema, forbidden HOW, taxonomy, pack
validity, split integrity, negative contamination, continuity isolation,
DecisionContext integrity, provider separation, objective-completion semantics.

---

## Soft metrics (batch_2)

| Metric | Result |
|--------|--------|
| Scenario diversity | PASS (211 families, 2.53% concentration) |
| Capability balance | PASS |
| Domain balance | PASS (0 ambiguous) |
| Pack balance | PASS |
| Continuity volume | **PASS** (77/77/77) |
| Edge-case coverage | PASS |
| Split capability balance | WARN (train-only inspect/branch — P2) |
| Repetition | PASS |

---

## P0 / P1 / P2 / P3 (post TR-7)

### P0
None.

### P1
None.

### P2
- Overrepresented: `artifact.attachment`, `diagnostics.collect_logs`, `workspace.search`
- Train-only capabilities: `repository.inspect`, `repository.branch`

### P3
- Formatter user preamble is generic contract text
- `context` provider pack underrepresented in Development formatter outputs

---

## TR-6 → TR-7 P1 reproduction notes

1. **Continuity volume** — independently confirmed at 15/15/15 on batch_1.
2. **Ambiguous labels** — TR-6 reported **203** using narrow tokens
   (`odoo` / `manifest` / `addons/` only). TR-7 semantic classifier (includes
   `models/partner`, `action_confirm`, `res.partner`, `__manifest__`, etc.)
   finds **304** label/cue mismatches on the same corpus. Discrepancy is
   expected: TR-6 under-counted cue-bearing Odoo examples as “ambiguous” when
   they lacked the narrow tokens, and missed some reverse mismatches.
   TR-7 corrected **303** unjustified `odoo` labels → generic, set **1**
   cue-bearing unlabeled record → `odoo`, quarantined **0**.

---

## Run

```bash
cd aiodoo-training
PYTHONPATH=. python3 -m aiodoo_training.system_training_contract.evaluation.cli \
  --corpus ../aiodoo-datasets/datasets/fp2/controlled_batch_2

PYTHONPATH=. python3 -m aiodoo_training.system_training_contract.generators.tr7_batch
```

---

## Exact next phase (do not start automatically)

Controlled adapter training may be authorized separately (Base Chat 1).  
Do **not** start LoRA / QLoRA / Colab / foundation downloads without that
authorization. Do **not** start TR-8 automatically.
