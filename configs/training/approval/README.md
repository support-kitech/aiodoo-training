# approval — Production Training Configuration

> **Dataset note:** v1.0.0 currently ships 1 approval record — expand in a future dataset release.
**Training ID:** `approval`  
**Published adapter:** `aiodoo-approval`  
**Progressive stage:** 7 / 8  
**Input adapter:** `aiodoo-conversation`  
**Dataset:** `datasets/v1.0.0/approval_dataset.jsonl` (1 records)

## Invoke

```bash
python train.py --config configs/training/approval/experiment.yaml
```

Colab:

```python
TRAINING_ID = "approval"
experiment = ExperimentStore(workspace=workspace).load(TRAINING_ID)
```

## Prerequisites

1. Prior product adapter published at `models/adapters/aiodoo-conversation/`
2. Dataset release `v1.0.0` present under workspace `datasets/v1.0.0/`
3. Base model `Qwen/Qwen3-8B` available via ModelStore / `AIODOO_COLAB_MODEL_PATH`

## Drive outputs

| Role | Path |
|------|------|
| Checkpoints | `training/cache/approval/checkpoints/` |
| Published adapter | `models/adapters/aiodoo-approval/` |
| Exports | `models/exports/aiodoo-approval/` |
| Run metadata | `experiments/approval/` |
