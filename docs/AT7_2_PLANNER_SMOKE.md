# AT-7.2 — Controlled Planner Skill Adapter Path Smoke

**Living.** Follows AT-7.1 (`REASONING_COVERAGE_MIXED`).  
**PATH SMOKE only.** Does **not** certify Planner. Does **not** full-train.  
Does **not** train Conversation / Approval / Evaluation. Does **not** merge.

---

## Verdict

**AT7_2_TRAINING_PATH_PASS**

| Layer | Value |
|-------|-------|
| DATA | native Planner **657**; Reasoning-pack **580** |
| SMOKE | train **16** / val **4** (isolated subset) |
| TRAINING | **4** steps; `train_loss=1.798` (finite) |
| EXPORT | real PEFT; reload + finite logits OK |

---

## Corpus (immutable)

| Item | Value |
|------|-------|
| Version | `fp2-controlled-2.0.0-tr7` |
| Checksum before/after | `728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227` |
| Native Planner | 657 (524 / 62 / 71) |
| Reasoning-pack Planner | 580 (A≡B) |
| Filter | `provider_capability==planner` AND `dataset_type==planner` |

---

## Smoke selection

Source: `pack_reasoning.jsonl` · strict subset of authoritative train/val IDs.  
Deterministic quotas by `record_type` + eng/decision round-robin + unique fingerprints.

| record_type | train smoke |
|-------------|------------:|
| planning_decision | 4 |
| engineering_feedback | 4 |
| decision_context | 4 |
| loop_decision | 4 |

`engineering_state` is **Development-only** in mapping — correctly absent from
Reasoning-pack smoke (not fabricated).

Duplicate-family presence: **4** selected records belong to corpus-level
continuity duplicate groups; smoke itself has **unique** fingerprints (no
intra-smoke dups).

Config: `configs/training/at7_planner_smoke/experiment.yaml`  
Prepare: `scripts/at72_prepare_planner_smoke.py`  
Outputs: `artifacts/at7_planner/smoke/`

---

## Training proof

| Check | Result |
|-------|--------|
| Foundation | local Qwen2.5-Coder-3B-Instruct (`AIODOO_HF_LOCAL_FILES_ONLY=1`) |
| Plane | **reasoning** / provider **planner** (independent) |
| Device | CPU |
| LoRA | r=8, α=16, dropout=0.05, q/k/v/o_proj |
| Step losses | step2=**1.832**, step4=**1.764** |
| Final loss | **1.798** |
| Checkpoint | `artifacts/at7_planner/smoke/checkpoints/checkpoint-4` |
| Adapter | `adapter_model.safetensors` (**14783648** bytes) |
| Reload / logits | PASS |
| Coding/Repair/Context loaded | **No** |

Provenance: `system_training_contract_version=1.0.0`,
`provider_contract_version=1.0.0`, planner / reasoning / checksum match.

---

## Not claimed

Production readiness · certification · full 524-train · Conversation/Approval/Evaluation · merge

---

## Next

**STOP.** Await authorization. Recommended later options: full Planner controlled
train, or Reasoning sparse-skill data phases (conversation/approval/evaluation).
