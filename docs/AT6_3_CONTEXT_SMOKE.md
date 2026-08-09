# AT-6.3 — Controlled Context Skill Adapter Path Smoke

**Living.** Follows AT-6.2.1 (`CONTEXT_CORPUS_READY`).  
**PATH SMOKE only.** Does **not** certify Context. Does **not** full-train.  
Does **not** modify corpora. Does **not** load Coding/Repair. Does **not** merge.

---

## Verdict

**AT6_3_TRAINING_PATH_PASS**

| Layer | Value |
|-------|-------|
| DATA | `fp2-context-controlled-1.0.0` — 261 native (immutable) |
| SMOKE | train **16** / val **4** (derived subset only) |
| TRAINING | **4** real steps; `train_loss=3.099` (finite) |
| EXPORT | real PEFT adapter; reload + finite logits OK |

---

## Corpus (immutable input)

| Item | Value |
|------|-------|
| Version | `fp2-context-controlled-1.0.0` |
| Checksum | `78e3b464d51b7e15912ca9aabc4ce65a579c4fbaddf5fa917609ac63c50ead87` |
| Size | 261 |
| Filter | `metadata.provider_capability==context` AND `dataset_type==context` |

Checksum before = after. Also unchanged: `controlled_batch_2`, AT-6.2 fixtures (26), legacy `context_v1_0.jsonl` (50161).

---

## Smoke selection

Deterministic quotas by `record_type`, then capability round-robin
(`sha256(record_id)`). Strict subset of Context train/val splits.

Train distribution: intent 8 / obs 8; search/navigate/read/inspect **4 each**;
Odoo 7 / generic 9.

Config: `configs/training/at6_context_smoke/experiment.yaml`  
Prepare: `scripts/at63_prepare_context_smoke.py`  
Outputs: `artifacts/at6_context/smoke/`

---

## Training proof

| Check | Result |
|-------|--------|
| Foundation | local `Qwen/Qwen2.5-Coder-3B-Instruct` (`AIODOO_HF_LOCAL_FILES_ONLY=1`) |
| Device | CPU (`cuda_available=false`, torch `2.12.0+cu130`) |
| Steps | 4 |
| Step losses (logged) | step2=2.649, step4=3.548 |
| Final train loss | 3.099 |
| Checkpoint | `artifacts/at6_context/smoke/checkpoints/checkpoint-4` |
| Adapter weight | `adapter_model.safetensors` (**14783648** bytes) |
| Reload | `PeftModel.from_pretrained` OK |
| Finite logits | PASS |
| Coding/Repair loaded | **No** |

Provenance (`artifacts/fp2_provenance.json`): distinguishes
`provider_contract_version` vs `system_training_contract_version` (**1.0.0**).

---

## Not claimed

- Production-ready / certified Context adapter  
- Full 198-train Context training  
- Legacy 50k projection  
- Merge with Coding / Repair / Execution  

---

## Next

**STOP.** Await authorization before full Context train, certification, or legacy projection.
