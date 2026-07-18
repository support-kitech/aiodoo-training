# evaluation — Production Training Configuration

> **Dataset note:** v1.0.0 currently ships 1 evaluation record — expand in a future dataset release.
**Training ID:** `evaluation`  
**Published adapter:** `aiodoo-evaluation`  
**Progressive stage:** 8 / 8  
**Input adapter:** `aiodoo-approval`  
**Dataset:** `datasets/v1.0.0/evaluation_dataset.jsonl` (1 records)

## Invoke

```bash
python train.py --config configs/training/evaluation/experiment.yaml
```

Colab:

```python
TRAINING_ID = "evaluation"
experiment = ExperimentStore(workspace=workspace).load(TRAINING_ID)
```

## Prerequisites

1. Prior product adapter published at `models/adapters/aiodoo-approval/`
2. Dataset release `v1.0.0` present under workspace `datasets/v1.0.0/`
3. Base model `Qwen/Qwen3-8B` available via ModelStore / `AIODOO_COLAB_MODEL_PATH`

## Drive outputs

| Role | Path |
|------|------|
| Checkpoints | `training/cache/evaluation/checkpoints/` |
| Published adapter | `models/adapters/aiodoo-evaluation/` |
| Exports | `models/exports/aiodoo-evaluation/` |
| Run metadata | `experiments/evaluation/` |
