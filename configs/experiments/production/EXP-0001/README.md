# EXP-0001 — First Production AIODOO Training Experiment

**Status:** Canonical production experiment  
**Experiment ID:** `EXP-0001`  
**Dataset version:** `v1.0.0`  
**Base model:** `Qwen/Qwen3-8B` (Hugging Face)  
**Hardware profile:** Google Colab Tesla T4 (16 GB)

This directory is the **source of truth** for the first production AIODOO model.

---

## Purpose

Train a progressive Odoo-engineering language model that specialises through
eight dataset stages (coding → … → evaluation). Each stage produces a PEFT
adapter that becomes the input adapter for the next stage.

`EXP-0001` must remain:

- executable via `python train.py --config …/experiment.yaml`
- launchable from **aiodoo-colab** with workspace overlays (`AIODOO_COLAB_*`)
- free of host-specific absolute paths (no hardcoded Linux / Drive paths)

---

## Directory structure

```text
configs/experiments/production/EXP-0001/
├── README.md            ← this file
├── experiment.yaml      ← root config (includes fragments)
├── dataset.yaml         ← dataset version + progressive stage table
├── model.yaml           ← base model identity + future EXP-0002 note
├── training.yaml        ← production trainer / QLoRA / packing / ckpt
├── evaluation.yaml      ← smoke + production gates
└── export.yaml          ← adapter / merge / fingerprints / future GGUF
```

---

## Training order (progressive)

```text
Stage 1  Base Qwen3-8B
            ↓  coding_v1_0.jsonl
         Adapter v1  (EXP-0001-stage01-coding)
            ↓
Stage 2  Resume Adapter v1
            ↓  planner_v1_0.jsonl
         Adapter v2  (EXP-0001-stage02-planner)
            ↓
Stage 3  Resume Adapter v2
            ↓  execution_dataset.jsonl
         Adapter v3
            ↓
Stage 4  Resume → repair_v1_0.jsonl        → Adapter v4
Stage 5  Resume → context_v1_0.jsonl       → Adapter v5
Stage 6  Resume → conversation_dataset.jsonl → Adapter v6
Stage 7  Resume → approval_dataset.jsonl   → Adapter v7
Stage 8  Resume → evaluation_dataset.jsonl → Adapter v8 (final)

Export: Adapter v8 (+ optional merge) → models/exports/EXP-0001/
```

**Rule:** the output adapter of stage *N* is the `resume_from` /
input adapter of stage *N+1*. Do not skip stages.

How to advance a stage:

1. Confirm Stage *N* checkpoints and adapter export exist.
2. In `dataset.yaml`, set `datasets[0]` to the next stage file / type.
3. In `training.yaml`, set `checkpointing.resume_from` to Stage *N* final adapter
   or checkpoint directory (workspace-relative / Colab-overlaid path).
4. Re-run `train.py` with the same `experiment.yaml`.
5. Export Stage *N+1* adapter with naming `EXP-0001-stage{NN}-{name}`.

---

## Hardware assumptions

| Item | Value | Rationale |
|------|-------|-----------|
| GPU | Tesla T4 16 GB | Default Colab GPU |
| Precision | fp16 + QLoRA 4-bit | T4 lacks production bf16; 8B needs 4-bit |
| Context | 2048 tokens | Longest stable length under packing on T4 |
| Micro-batch | 1 | Avoids OOM |
| Grad accumulation | 16 | Effective batch 16 for quality |
| Grad checkpointing | on | Saves activation memory |
| Flash Attention 2 | optional | Enable only if host provides kernels |

---

## Dataset version

- **Version:** `v1.0.0`
- **Workspace root:** `<workspace>/datasets/v1.0.0/`
- **Files (stage order):**

| Stage | File | Type |
|------:|------|------|
| 1 | `coding_v1_0.jsonl` | coding |
| 2 | `planner_v1_0.jsonl` | planner |
| 3 | `execution_dataset.jsonl` | execution |
| 4 | `repair_v1_0.jsonl` | repair |
| 5 | `context_v1_0.jsonl` | context |
| 6 | `conversation_dataset.jsonl` | conversation |
| 7 | `approval_dataset.jsonl` | approval |
| 8 | `evaluation_dataset.jsonl` | evaluation |

Paths are **workspace-relative**. aiodoo-colab resolves
`dataset_version` → `workspace.datasets / v1.0.0` and sets
`AIODOO_COLAB_DATASET_PATH`.

---

## Base model

| Field | Value |
|-------|--------|
| Hub id | `Qwen/Qwen3-8B` |
| Provider | Hugging Face |
| Workspace cache | `models/base/Qwen__Qwen3-8B/` |
| Backend | `hf_causal` |

**Future:** `EXP-0002` → `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
(see `model.yaml`).

Colab sets `AIODOO_COLAB_MODEL_PATH` to the local snapshot — never embed Drive paths in YAML.

---

## Output locations (workspace)

| Artifact | Path |
|----------|------|
| Adapters | `models/adapters/EXP-0001/` |
| Checkpoints | `models/adapters/EXP-0001/checkpoints/` |
| Merged | `models/merged/EXP-0001/` |
| Exports | `models/exports/EXP-0001/` |
| Logs | `logs/EXP-0001/` |
| Metrics | `logs/EXP-0001/metrics/` |
| Tracking | `artifacts/tracking/EXP-0001/` (repo-local) / overlaid logs |

Colab overlays: `AIODOO_COLAB_ADAPTER_OUTPUT`, `AIODOO_COLAB_CHECKPOINTS_OUTPUT`,
`AIODOO_COLAB_MERGED_OUTPUT`, `AIODOO_COLAB_EXPORT_OUTPUT`, `AIODOO_COLAB_LOGS_OUTPUT`,
`AIODOO_COLAB_METRICS_OUTPUT`.

---

## Checkpoint strategy

| Setting | Value | Why |
|---------|-------|-----|
| `save_steps` | 200 | ~resume frequency under Colab preempt risk |
| `save_total_limit` | 3 | Keep last three; prune older automatically |
| `save_on_failure` | true | Capture crash state |
| `validate_on_load` | true | Fingerprint-safe resume |
| Retention | `keep_last` | Match limit policy |
| Adapter naming | `EXP-0001-stage{NN}-{name}` | Stable progressive chain |

**Cleanup:** relying on `save_total_limit` plus Drive quota reviews; do not
delete the **final** adapter of a completed stage until the next stage exports.

---

## Resume strategy

- Policy: **strict** (fingerprint mismatch blocks resume).
- Stage 1: `resume_from: null`.
- Stage N>1: `resume_from` → Stage N−1 final adapter/checkpoint.
- After Colab disconnect: resume latest checkpoint under the stage checkpoint dir
  without changing stage identity.

---

## Logging strategy

- Trainer `logging_steps: 10`
- Tracking backend: `local_jsonl` (`tracker_type: local_jsonl`)
- Metric history: `artifacts/checkpoints/EXP-0001/metrics/history.jsonl`
  (Colab remaps under `logs/EXP-0001/metrics/`)
- stderr operational logs from `aiodoo_training` CLI (`logger.exception` on failure)

### Progress monitoring

| Signal | Where | Meaning |
|--------|-------|---------|
| Training loss | stdout / metrics history / tracker | Primary optimisation signal |
| Learning rate | logs / Trainer state | Cosine schedule after warmup |
| Epoch / step | Pipeline stage messages + Trainer | Stage progress |
| Current dataset | `dataset.yaml` active entry + curriculum stage name | Active stage file |
| Checkpoint creation | checkpoint dir mtime / trainer save logs | Resume points |
| Resume | `resume_from` + startup log | Confirms restore |
| Logs | `logs/EXP-0001/` | Run transcript |
| TensorBoard | not primary | Optional if host enables `report_to=tensorboard`; default is local_jsonl |
| Trainer state | checkpoint `trainer_state`-equivalent / progress objects | Internal step/epoch |

Watch for: rising loss, repeated OOMs, or missing checkpoints for >2× `save_steps`.

---

## Evaluation strategy

- **Smoke:** after every stage (`max_examples` small) — catch broken adapters early.
- **Production:** after Stage 8 — acceptance thresholds in `evaluation.yaml`
  (`loss ≤ 2.5`, soft `token_accuracy ≥ 0.35`).
- Sample prompts listed in `evaluation.yaml` for notebook/human review.

---

## Export strategy

Always export:

1. PEFT adapter  
2. Manifest + fingerprints  
3. Tokenizer binding metadata  
4. Model card  

After Stage 8 pass:

5. Merged base+adapter under `models/merged/EXP-0001/`

**GGUF:** reserved under `models/exports/EXP-0001/gguf/` for a future
quantisation job — **not** part of tonight’s EXP-0001 export set.

---

## Expected outputs

After a successful Stage 1 (coding) run:

- Checkpoints under adapter checkpoint dir  
- Metrics history JSONL  
- Structured `ExecutionResult` on stdout (`success`, paths, duration)  
- Tracking records when `local_jsonl` is active  

After Stage 8 + export:

- Final adapter `EXP-0001-stage08-evaluation`  
- Merged model directory  
- Export bundle + fingerprints  

---

## How to run

### From aiodoo-training (repository root)

```bash
python train.py --config configs/experiments/production/EXP-0001/experiment.yaml
```

Ensure datasets exist under the workspace datasets version root and (for GPU
runs) set Colab/env overlays for model and outputs.

### From aiodoo-colab

1. Mount Drive / prepare workspace (`AIODOO/`).
2. Place datasets at `AIODOO/datasets/v1.0.0/`.
3. Ensure `Qwen/Qwen3-8B` is cached under `AIODOO/models/base/Qwen__Qwen3-8B/`.
4. Clone/update `aiodoo-training` under `AIODOO/training/aiodoo-training`
   (canonical EXP-0001 configs live there and are auto-discovered).
5. Optional: mirror configs under `AIODOO/experiments/EXP-0001/config/` for
   Drive-local overrides; otherwise Colab loads the training-repo canonical set.
6. Launch `notebooks/01_train.ipynb` with `EXPERIMENT_ID = "EXP-0001"`.

Colab sets `AIODOO_COLAB_*` path overlays and invokes:

```text
python train.py --config <aiodoo-training>/configs/experiments/production/EXP-0001/experiment.yaml
```

### Smoke test

See [SMOKE.md](./SMOKE.md) — same model/training/export, smaller JSONL only.

---

## Quality profile summary (`training.yaml`)

| Knob | Production choice |
|------|-------------------|
| Adaptation | QLoRA rank 16 / alpha 32 / full attn+MLP targets |
| Precision | fp16 + 4-bit load + activation checkpointing |
| Context | 2048 + concat packing |
| Effective batch | 16 |
| Optim / sched | AdamW + cosine + 3% warmup |
| LR | 1e-4 |
| Curriculum | sequential 8 stages |
| Save | every 200 steps, keep 3 |

Chosen for **maximum practical quality on Tesla T4**, not for synthetic peak
numbers that OOM mid-stage.
