# AT-7.6 — Controlled Evaluation Skill Adapter Path Smoke

**Living.** Follows AT-7.5 (`EVALUATION_CORPUS_READY`).  
**PATH SMOKE only.** Does **not** certify Evaluation. Does **not** full-train.  
Does **not** train Conversation / Approval / Planner. Does **not** merge.

---

## Verdict

**AT7_6_TRAINING_PATH_PASS**

| Layer | Value |
|-------|-------|
| DATA | native Evaluation **252** (`evaluation_judgment`) |
| SMOKE | train **16** / val **4** (isolated subset) |
| TRAINING | **4** steps; `train_loss=2.733` (finite) |
| EXPORT | real PEFT; reload + finite logits OK |

---

## Corpus (immutable)

| Item | Value |
|------|-------|
| Version | `fp2-evaluation-controlled-1.0.0` |
| Checksum before/after | `764dba2849519c2b3cf1f5ff24acb84c644f3506b99dbc958762e470310e0883` |
| Native | 252 (200 / 48 / 4) |
| Filter | `provider_capability==evaluation` AND `dataset_type==evaluation` AND `record_type==evaluation_judgment` |
| Source | `pack_evaluation.jsonl` |

---

## Smoke selection

Prepare: `scripts/at76_prepare_evaluation_smoke.py`  
Config: `configs/training/at7_evaluation_smoke/experiment.yaml`  
Outputs: `artifacts/at7_evaluation/smoke/`

Deterministic quotas by verdict → round-robin `candidate_category`; unique
fingerprints; ≤1 record per `scenario_family`; disjoint from corpus test
families.

| Metric | Train smoke |
|--------|-------------|
| Verdicts | pass 6 / fail 5 / inconclusive 5 |
| Categories | planner 3, conversation 3, approval 3, generic 3, coding 2, repair 1, execution 1 |
| Domains | covered (Odoo + generic) |
| Family leakage (train∩val∩test) | **0** |
| Intra-smoke duplicate fingerprints | **0** |

Note: `context` category is present in the corpus but not in this 16-row smoke
under the deterministic quota; representative coverage still spans 7/8
categories plus all verdicts and optional-field patterns in the broader subset
selection.

---

## Training proof

| Check | Result |
|-------|--------|
| Foundation | local Qwen2.5-Coder-3B-Instruct (`AIODOO_HF_LOCAL_FILES_ONLY=1`) |
| Plane | **reasoning** / provider **evaluation** (independent) |
| Device | CPU |
| LoRA | r=8, α=16, dropout=0.05, q/k/v/o_proj |
| Step losses | step2=**2.845**, step4=**2.622** |
| Final loss | **2.733** |
| Checkpoint | `artifacts/at7_evaluation/smoke/checkpoints/checkpoint-4` |
| Adapter | `adapter_model.safetensors` (**14783648** bytes) |
| Reload / logits | PASS |
| Coding/Repair/Context/Planner/Conversation/Approval loaded | **No** |

Provenance: `system_training_contract_version=1.0.0`,
`provider_contract_version=1.0.0`, evaluation / reasoning / checksum match,
`record_type=evaluation_judgment`.

---

## Immutability

| Artifact | Status |
|----------|--------|
| Evaluation corpus | unchanged |
| controlled_batch_2 | `728d9bad…f227` unchanged |
| Conversation 232 | unchanged |
| Approval 162 | unchanged |
| Planner / Context / Coding / Repair / Execution | not modified |
| Legacy `evaluation_dataset.jsonl` | not projected |

`LEGACY_PROJECTION = NOT PERFORMED`

---

## Not claimed

Production readiness · certification · full 200-train · quality of judgments ·
merge · Conversation/Approval train

---

## Next

**STOP.** Await authorization.

Recommended later options (not auto-started):

1. Conversation path smoke  
2. Approval path smoke  
3. Resource-controlled full Evaluation / Planner train  
4. Optional Evaluation legacy → FP2 projection phase  
5. Certification / packaging (separate)

Do not start AT-8 automatically.
