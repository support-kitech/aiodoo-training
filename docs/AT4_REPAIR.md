# AT-4 — Controlled Repair Skill Adapter

**Living.** Follows AT-3 / AT-3.1 (`AT3_TRAINING_PATH_PASS`, `CODING_COVERAGE_CONFIRMED`).  
Independent Repair provider adapter on the proven FP2 → HF/PEFT path.  
**Does not** certify quality. **Does not** merge with Coding. **Does not** train other skills.

---

## Verdict

**AT4_TRAINING_PATH_PASS**

| Check | Status |
|-------|--------|
| Repair filter (`provider_capability=repair`) | **OK** — 42 eligible |
| Split `fp2-split-1.0.0` | **OK** — 28 / 7 / 7; no leakage |
| Local Qwen foundation | **OK** |
| Real LoRA + 28 train steps | **OK** — finite `train_loss=1.873` |
| Checkpoints | **OK** — `checkpoint-14`, `checkpoint-28` |
| Real PEFT export | **OK** — ~14.8 MB weights |
| Provenance | **OK** — STC `1.0.0`, repair, checksum |
| Independent reload + logits | **OK** (no Coding adapter) |
| `controlled_batch_2` unchanged | **OK** |
| AT-3 / AT-2.1 artifacts untouched | **OK** |

**Not claimed:** production readiness / certification. Population is only **42** labeled repair records.

---

## Selection

| Field | Value |
|-------|-------|
| Filter | `metadata.provider_capability==repair` (and `dataset_type==repair`, identical) |
| Eligible | 42 |
| Train / val / test | 28 / 7 / 7 |
| Record types | CI 13, EWU 13, observation 14, engineering_feedback 2 |
| Steps | 28 × batch 1 × 1 epoch |

Derived only under `artifacts/at4_repair/data/`.

---

## Config / outputs

- Config: `configs/training/at4_repair/experiment.yaml`
- Checkpoints: `artifacts/at4_repair/checkpoints/`
- Export: `artifacts/at4_repair/export/bundle-exp_7537f8af0aa64a6e-337a7c8baecd/`
- Result: `artifacts/at4_repair/train_result.json`

Production `configs/training/repair/*` (legacy / DeepSeek / Colab) was **not** used.

---

## Next

STOP. Await authorization for AT-5 (next independent skill).  
Do not merge Coding+Repair. Do not certify.
