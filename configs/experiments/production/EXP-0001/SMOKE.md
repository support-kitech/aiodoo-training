# EXP-0001 Smoke Test Procedure

Use the **same** production model, training knobs, pipeline, and export settings.
Only the **dataset file size** changes.

## Goal

Validate the full Colab → `train.py` → checkpoint → adapter path on Tesla T4
before starting Stage 1 on the full `coding_v1_0.jsonl`.

## Prerequisites

1. Drive workspace prepared (`AIODOO/…`)
2. `aiodoo-training` cloned under `AIODOO/training/aiodoo-training`
3. `Qwen/Qwen3-8B` cached at `AIODOO/models/base/Qwen__Qwen3-8B/`
4. GPU runtime with `pip install -r requirements/train.txt`

## Create a 50-record smoke set (no training-code changes)

From a machine that has the full Stage 1 JSONL:

```bash
DATASET_ROOT="$AIODOO_WORKSPACE/datasets/v1.0.0"   # or your Drive path
SRC="$DATASET_ROOT/coding_v1_0.jsonl"
SMOKE_DIR="$DATASET_ROOT/smoke"
mkdir -p "$SMOKE_DIR"
head -n 50 "$SRC" > "$SMOKE_DIR/coding_v1_0.jsonl"
```

100-record variant:

```bash
head -n 100 "$SRC" > "$SMOKE_DIR/coding_v1_0.jsonl"
```

Do **not** edit trainer code. Point the experiment at the smoke directory by
temporarily setting Colab dataset resolution:

### Option A — temporary dataset version overlay (recommended)

1. Place smoke file at:
   `AIODOO/datasets/v1.0.0-smoke/coding_v1_0.jsonl`
2. In Drive experiment `config/dataset.yaml` **or** for a one-off notebook cell
   before launch, ensure `dataset_version: "v1.0.0-smoke"` while keeping the
   same relative filename `coding_v1_0.jsonl`.
3. Run the notebook as usual (`EXPERIMENT_ID = "EXP-0001"`).
4. Restore `dataset_version: "v1.0.0"` after smoke passes.

### Option B — replace-only for smoke host

Copy smoke JSONL over a private host copy of `coding_v1_0.jsonl` used only for
smoke. Never commit truncated production datasets.

## Success criteria (smoke)

| Check | Expect |
|-------|--------|
| `train.py` exit | `0` |
| `ExecutionResult.success` | `True` |
| Checkpoint dir | Non-empty under `models/adapters/EXP-0001/checkpoints/` |
| Loss logged | Finite loss entries every `logging_steps` |
| No OOM | Process completes without CUDA OOM |
| Adapter export | Adapter files under adapter output when export stage runs |

## After smoke

1. Restore full `v1.0.0` datasets.
2. Clear smoke checkpoints if you do not want them mixed with Stage 1.
3. Start Stage 1 production on full `coding_v1_0.jsonl` with `resume_from: null`.
