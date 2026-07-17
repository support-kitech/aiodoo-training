# Production Smoke Test — EXP-0001

**Status:** Pre-smoke contract (Drive workspace `drive_v1`)

This document is the authoritative procedure for the first end-to-end production
smoke test. All paths follow `aiodoo-training` `ArtifactOutputLayout` — the
single production contract shared by Colab, training, validation, and models.

## Canonical Drive layout

```text
AIODOO/
├── datasets/
│   └── v1.0.0/                         # smoke dataset version
├── training/
│   ├── aiodoo-training/                # cloned training repo (source only)
│   └── cache/
│       └── EXP-0001/
│           └── checkpoints/            # runtime checkpoints (transient)
│               └── checkpoint-*/
├── models/
│   ├── adapters/
│   │   └── EXP-0001/                   # published adapter + artifact.json
│   ├── merged/
│   │   └── EXP-0001/                   # optional (when export merges)
│   └── exports/
│       └── EXP-0001/
│           └── bundle-EXP-0001-*/      # export bundle (aiodoo-models contract)
├── experiments/
│   └── EXP-0001/
│       ├── config/                     # config snapshot
│       ├── metrics/                    # history.jsonl
│       ├── logs/tracking/              # run tracking JSONL
│       └── summary.json                # experiment summary
└── experiments/EXP-0001/config/        # read-only experiment YAML (Drive)
```

Base Hugging Face models live on Colab local SSD (not Drive):

```text
/content/aiodoo-model-cache/Qwen__Qwen3-8B/
└── artifact.json                       # written by training finalize
```

## Prerequisites

1. Google Drive mounted with AIODOO workspace at e.g.
   `/content/drive/MyDrive/colab_notebooks/AIODOO`
2. `aiodoo-training` cloned under `AIODOO/training/aiodoo-training/`
3. Smoke dataset at `AIODOO/datasets/v1.0.0/`
4. `aiodoo-colab` Phase 5+ orchestration available
5. `aiodoo-validation` v1.0.0 installed for post-training validation

## Step 1 — Training (Colab)

```python
from config import load_config
from workspace import Workspace, ensure_workspace_layout
from experiments import ExperimentStore
from models import ModelStore
from trainer import build_training_context, run_training

config = load_config(auto_mount_drive=True)
ws = Workspace.from_config(config)
ensure_workspace_layout(ws)

experiment = ExperimentStore(workspace=ws).load("EXP-0001")
ModelStore(workspace=ws, model_id=experiment.model_id).ensure()

context = build_training_context(ws, experiment)
result = run_training(context)
```

Colab sets these environment variables for `train.py`:

| Variable | Purpose |
|----------|---------|
| `AIODOO_WORKSPACE_ROOT` | **Required** — canonical Drive workspace root |
| `AIODOO_COLAB_MODEL_PATH` | Local SSD base model directory |
| `AIODOO_COLAB_DATASET_PATH` | Dataset version root (`datasets/v1.0.0`) |

Training derives all artifact paths from `AIODOO_WORKSPACE_ROOT`. Individual
`AIODOO_COLAB_*_OUTPUT` path hints are **not** used.

### Post-training verification

```bash
export AIODOO_WORKSPACE_ROOT=/content/drive/MyDrive/colab_notebooks/AIODOO
EXP=EXP-0001

# Runtime checkpoint
ls "$AIODOO_WORKSPACE_ROOT/training/cache/$EXP/checkpoints/checkpoint-"*

# Published adapter (inference-only + validation handoff)
ls "$AIODOO_WORKSPACE_ROOT/models/adapters/$EXP/"
test -f "$AIODOO_WORKSPACE_ROOT/models/adapters/$EXP/artifact.json"
test -f "$AIODOO_WORKSPACE_ROOT/models/adapters/$EXP/adapter_model.safetensors"
! test -f "$AIODOO_WORKSPACE_ROOT/models/adapters/$EXP/rng.json"

# Export bundle
ls "$AIODOO_WORKSPACE_ROOT/models/exports/$EXP/bundle-"*/

# Experiment summary
test -f "$AIODOO_WORKSPACE_ROOT/experiments/$EXP/summary.json"

# Base model validation handoff
test -f /content/aiodoo-model-cache/Qwen__Qwen3-8B/artifact.json
```

## Step 2 — Validation (immediate, no manual bridge files)

```bash
aiodoo-validation validate \
  --profile coding \
  --base-model /content/aiodoo-model-cache/Qwen__Qwen3-8B \
  --adapter "$AIODOO_WORKSPACE_ROOT/models/adapters/EXP-0001" \
  --execution-tier smoke \
  --odoo-versions 18
```

Validation resolves `artifact.json` in both the base model and adapter
directories. No manual bridge files are required.

## Step 3 — Cleanup (optional, safe)

```bash
cd "$AIODOO_WORKSPACE_ROOT/training/aiodoo-training"
python cleanup_artifacts.py "$AIODOO_WORKSPACE_ROOT"        # dry-run
python cleanup_artifacts.py "$AIODOO_WORKSPACE_ROOT" --delete  # runtime only
```

Cleanup never deletes published adapters, merged models, export bundles,
experiment summaries, or validation reports.

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `AIODOO_WORKSPACE_ROOT is not set` | Missing workspace env | Colab must set `AIODOO_WORKSPACE_ROOT` |
| Validation `Missing artifact.json` | Publish did not run | Ensure training pipeline completed; check finalize logs |
| Adapter contains `rng.json` | Old publish logic | Re-run training with current aiodoo-training |
| Paths under repo clone | Workspace root missing | Set `AIODOO_WORKSPACE_ROOT`; never write to repo |

## What to freeze after smoke succeeds

1. `ArtifactOutputLayout` path contract
2. `artifact.json` validation handoff schema
3. `AIODOO_WORKSPACE_ROOT` requirement for `drive_v1`
4. Export bundle manifest format (including `dataset_version`)
5. `aiodoo-validation` v1.0.0 public API

See also: [artifact_output_pipeline.md](artifact_output_pipeline.md)
