# AT-3 — Controlled Coding Skill Adapter

**Living.** Follows AT-2.1 (`SMOKE_PASS`).  
First production-path independent skill adapter on the proven FP2 → HF/PEFT chain.  
**Does not** certify quality. **Does not** train other skills. **Does not** merge.

---

## Verdict

**AT3_TRAINING_PATH_PASS**

| Check | Status |
|-------|--------|
| Coding FP2 filter (`provider_capability=coding`) | **OK** — 26 eligible |
| Split `fp2-split-1.0.0` | **OK** — 22 / 1 / 3; no family leakage |
| Local Qwen foundation | **OK** |
| Real LoRA + 22 train steps | **OK** — finite `train_loss=3.233` |
| Checkpoints | **OK** — `checkpoint-11`, `checkpoint-22` |
| Real PEFT export | **OK** — `adapter_model.safetensors` ~14.8 MB |
| Provenance | **OK** — System Training Contract `1.0.0` ≠ provider contract |
| Adapter reload + logit sanity | **OK** |
| `controlled_batch_2` unchanged | **OK** |
| AT-2.1 smoke untouched | **OK** |

---

## Dataset selection

Source (immutable): `aiodoo-datasets/datasets/fp2/controlled_batch_2/`

Filter (from Training Contract / pack metadata):

`metadata.provider_capability == "coding"` AND `dataset_type == "coding"`

joined to `splits.jsonl` via `record_id` (`fp2-split-1.0.0`).

Derived only under `artifacts/at3_coding/data/` (does not modify batch_2).

---

## Config

`configs/training/at3_coding/experiment.yaml`

Isolated from production `configs/training/coding/*` (Colab / DeepSeek / QLoRA).  
Uses AT-2.1-proven local Qwen + CPU LoRA path.

---

## Not claimed

- Adapter certification / production readiness  
- Complete Development product  
- Other seven skill adapters  

---

## Next

STOP. Await authorization for AT-4 (next skill, likely `repair`) or further coding scale-up.
