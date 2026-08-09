# AT-7.7 — Controlled Conversation Skill Adapter Path Smoke

**Living.** Follows AT-7.3 (`CONVERSATION_CORPUS_READY`) and AT-7.6.  
**PATH SMOKE only.** Does **not** certify Conversation. Does **not** full-train.  
Does **not** train Approval / Evaluation / Planner. Does **not** merge.
Does **not** project legacy `conversation_dataset.jsonl`.

---

## Verdict

**AT7_7_TRAINING_PATH_PASS**

| Layer | Value |
|-------|-------|
| DATA | native Conversation **232** (`decision_context` + `loop_decision`/clarify) |
| SMOKE | train **16** / val **4** (isolated subset) |
| TRAINING | **4** steps; `train_loss=2.992` (finite) |
| EXPORT | real PEFT; reload + finite logits OK |

---

## Corpus (immutable)

| Item | Value |
|------|-------|
| Version | `fp2-reasoning-sparse-1.0.0` |
| Checksum before/after | `488b3a7576071c875c32e277c49562bb9c472904e32b12a1b98fcf6558da9de3` |
| Native | 232 (180 / 28 / 24) |
| Families | `decision_context`, `loop_decision` (clarify) |
| Filter | `provider_capability==conversation` AND `dataset_type==conversation` |
| Source | `pack_reasoning.jsonl` |

---

## Smoke selection

Prepare: `scripts/at77_prepare_conversation_smoke.py`  
Config: `configs/training/at7_conversation_smoke/experiment.yaml`  
Outputs: `artifacts/at7_conversation/smoke/`

Deterministic quotas: train 8 `decision_context` + 8 `loop_decision`/clarify;
val 2+2; unique fingerprints; ≤1 record per `scenario_family`; disjoint from
corpus test families.

| Metric | Train smoke |
|--------|-------------|
| Record types | decision_context 8 / loop_decision 8 |
| Decision kinds | clarify 8 / none 8 |
| Domains | odoo 8 / generic 8 |
| Family leakage (train∩val∩test) | **0** |
| Intra-smoke duplicate fingerprints | **0** |

---

## Training proof

| Check | Result |
|-------|--------|
| Foundation | local Qwen2.5-Coder-3B-Instruct (`AIODOO_HF_LOCAL_FILES_ONLY=1`) |
| Plane | **reasoning** / provider **conversation** (independent) |
| Device | CPU |
| LoRA | r=8, α=16, dropout=0.05, q/k/v/o_proj |
| Step losses | step2=**3.086**, step4=**2.899** |
| Final loss | **2.992** |
| Checkpoint | `artifacts/at7_conversation/smoke/checkpoints/checkpoint-4` |
| Adapter | `adapter_model.safetensors` (**14783648** bytes) |
| Reload / logits | PASS |
| Coding/Repair/Context/Planner/Approval/Evaluation loaded | **No** |

Provenance: `system_training_contract_version=1.0.0`,
`provider_contract_version=1.0.0`, conversation / reasoning / checksum match.

---

## Immutability

| Artifact | Status |
|----------|--------|
| Conversation corpus | unchanged |
| controlled_batch_2 | `728d9bad…f227` unchanged |
| Approval 162 | unchanged |
| Evaluation 252 | unchanged |
| Planner / Context / Coding / Repair / Execution | not modified |
| Legacy `conversation_dataset.jsonl` | not projected |

`LEGACY_PROJECTION = NOT PERFORMED`

---

## Not claimed

Production readiness · certification · full 180-train · conversational quality ·
merge · Approval train · AT-7.8

---

## Next

**STOP.** Await authorization.

Recommended later options (not auto-started):

1. Approval path smoke  
2. Resource-controlled full Conversation / Evaluation / Planner train  
3. Optional Conversation legacy → FP2 projection phase  
4. Certification / packaging (separate)

Do not start AT-7.8 or AT-8 automatically.
