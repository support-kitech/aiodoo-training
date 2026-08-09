# AT-2 / AT-2.1 — FP2 Training Path Fixes + Local Smoke

**Living.** Follows AT-1 (`READY_WITH_REQUIRED_FIXES`).  
Implements FP2 pack loading, real HF PEFT export, and FP2 provenance.  
**Does not** certify adapters. **Does not** run full skill training.

---

## Verdict

**AT-2.1: SMOKE_PASS**

| Objective | Status |
|-----------|--------|
| A. FP2 TrainingExample loader | **DONE** — `record_format: fp2_training_example` |
| B. Real HF PEFT export | **DONE** — `HFExporter` writes real `adapter_model.*` + config |
| C. FP2 provenance on artifacts | **DONE** — System Training Contract ≠ provider contract |
| D. Tiny real FP2 smoke | **SMOKE_PASS** (AT-2.1) — local Qwen Development foundation |

AT-2 alone stopped at **MODEL_UNAVAILABLE** (DeepSeek FW0 not local; no download).  
AT-2.1 recovered using the verified local Development foundation without Hub download.

---

## Fixes (source)

### A — Loader

- `DatasetRef.record_format`: `protocol_v1` (default) | `fp2_training_example`
- `JsonlDatasetSource` identity path via `datasets/fp2_example_loader.py`
- Legacy Protocol V1 formatters unchanged

### B — PEFT export

- `infrastructure/huggingface/exporter.py` calls `save_pretrained`
- Requires non-empty `adapter_model.safetensors` / `.bin` + `adapter_config.json`
- Refuses stub models when `hf_peft` is selected
- `ExportManager` accepts PEFT **directories** (adapter_config + non-empty weights)

### C — Provenance

- `build_adapter_artifact_json` / `_export_bind_extra` add:
  - `provider_contract_version` (provider `aiodoo_contract`)
  - `system_training_contract_version`
  - `corpus_version`, `corpus_checksum`, `source_pack`, `split`, `foundation_model_id`, …

### D — AT-2.1 local foundation

- Isolated config: `configs/training/at2_fp2_smoke/experiment_at21.yaml`
- Explicit `model.local_path` → AIODOO Development Qwen weights
- `AIODOO_HF_LOCAL_FILES_ONLY=1` / `local_files_only` when `local_path` set
- TokenizeStage honors `packing.max_sequence_length` for tiny smoke

---

## Smoke artifacts

| Item | Location |
|------|----------|
| Smoke subset (8 examples) | `artifacts/at2_smoke/data/pack_development_smoke.jsonl` |
| AT-2.1 config | `configs/training/at2_fp2_smoke/experiment_at21.yaml` |
| AT-2 config (DeepSeek; env-blocked) | `configs/training/at2_fp2_smoke/experiment.yaml` |
| AT-2.1 outputs | `artifacts/at2_smoke/at21/` |
| Result | `artifacts/at2_smoke/at21/smoke_result.json` |

`controlled_batch_2` checksum unchanged:
`728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227`

### AT-2.1 smoke command

```bash
cd aiodoo-training
AIODOO_HF_LOCAL_FILES_ONLY=1 PYTHONPATH=. \
  python3 train.py --config configs/training/at2_fp2_smoke/experiment_at21.yaml
```

| Field | Value |
|-------|-------|
| Foundation id | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| Local path | `AIODOO/models/foundations/development/Qwen2.5-Coder-3B-Instruct` |
| Device | CPU (`cuda=false`) |
| Steps | 2 |
| Losses | ~5.655, ~3.329 (`train_loss` ~4.49, finite) |
| Adapter | `adapter_model.safetensors` (~14.8 MB) + `adapter_config.json` |
| Reload | `PeftModel.from_pretrained` OK |

---

## Remaining (non-blocking for smoke)

| Priority | Issue |
|----------|-------|
| P2 | HF checkpoint `restore` still application-level (AT-1) |
| P2 | Production coding `family: deepseek-coder` enum mismatch vs `ModelFamily.deepseek` |
| P3 | CLI resume/export polish |
| P3 | AT-2 DeepSeek smoke config still blocked until FW0 weights are local |

---

## Next

STOP after AT-2.1. Await Base Chat 1 authorization before AT-3 / controlled skill training.  
Do **not** merge adapters or certify production artifacts from this smoke.
