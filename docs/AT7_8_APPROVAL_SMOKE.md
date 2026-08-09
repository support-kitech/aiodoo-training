# AT-7.8 — Controlled Approval Skill Adapter Path Smoke

**Living.** Follows AT-7.3 (`APPROVAL_CORPUS_READY`) and AT-7.7.  
**PATH SMOKE only.** Does **not** certify Approval. Does **not** full-train.  
Does **not** train Conversation / Evaluation / Planner. Does **not** merge.
Does **not** project legacy Approval data.

---

## Verdict

**AT7_8_TRAINING_PATH_PASS**

| Layer | Value |
|-------|-------|
| DATA | native Approval **162** (`loop_decision`: approve/reject/modify) |
| SMOKE | train **16** / val **4** (isolated subset) |
| TRAINING | **4** steps; `train_loss=4.814` (finite) |
| EXPORT | real PEFT; reload + finite logits OK |

---

## Corpus (immutable)

| Item | Value |
|------|-------|
| Version | `fp2-reasoning-sparse-1.0.0` |
| Checksum before/after | `3e069403348203e3b6aec2ce0f31d2dc622c60146b0bbde6dbfee3134fdfbcb7` |
| Native | 162 (134 / 14 / 14) |
| Record type | `loop_decision` |
| Decision kinds | approve 81 / reject 43 / modify 38 |
| Filter | `provider_capability==approval` AND `dataset_type==approval` |
| Source | `pack_reasoning.jsonl` |

---

## Smoke selection

Prepare: `scripts/at78_prepare_approval_smoke.py`  
Config: `configs/training/at7_approval_smoke/experiment.yaml`  
Outputs: `artifacts/at7_approval/smoke/`

Deterministic quotas by decision_kind: train approve **6** / reject **5** /
modify **5**; val 2/1/1; unique fingerprints; ≤1 record per `scenario_family`;
disjoint from corpus test families.

| Metric | Train smoke |
|--------|-------------|
| Decision kinds | approve 6 / reject 5 / modify 5 |
| Domains | odoo 7 / generic 9 |
| Family leakage (train∩val∩test) | **0** |
| Intra-smoke duplicate fingerprints | **0** |

---

## Training proof

| Check | Result |
|-------|--------|
| Foundation | local Qwen2.5-Coder-3B-Instruct (`AIODOO_HF_LOCAL_FILES_ONLY=1`) |
| Plane | **reasoning** / provider **approval** (independent) |
| Device | CPU |
| LoRA | r=8, α=16, dropout=0.05, q/k/v/o_proj |
| Step losses | step2=**4.593**, step4=**5.034** |
| Final loss | **4.814** |
| Checkpoint | `artifacts/at7_approval/smoke/checkpoints/checkpoint-4` |
| Adapter | `adapter_model.safetensors` (**14783648** bytes) |
| Reload / logits | PASS |
| Coding/Repair/Context/Planner/Conversation/Evaluation loaded | **No** |

Provenance: `system_training_contract_version=1.0.0`,
`provider_contract_version=1.0.0`, approval / reasoning / checksum match,
`record_type=loop_decision`.

---

## Immutability

| Artifact | Status |
|----------|--------|
| Approval corpus | unchanged |
| controlled_batch_2 | `728d9bad…f227` unchanged |
| Conversation 232 | unchanged |
| Evaluation 252 | unchanged |
| Planner / Context / Coding / Repair / Execution | not modified |
| Legacy Approval dataset | not projected |

`LEGACY_PROJECTION = NOT PERFORMED`

---

## Not claimed

Production readiness · certification · full 134-train · Approval quality ·
merge · Conversation/Evaluation/Planner train

---

## Next

**STOP.** Await authorization.

Recommended later options (not auto-started):

1. Resource-controlled full Approval / Conversation / Evaluation / Planner train  
2. Optional Approval legacy → FP2 projection phase  
3. Certification / packaging (separate)

Do not start AT-7.9 or AT-8 automatically.
