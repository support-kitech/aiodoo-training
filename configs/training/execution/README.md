# execution — Production Training Configuration

**Training ID:** `execution`  
**Published adapter:** `aiodoo-execution`  
**Progressive stage:** 3 / 8  
**Input adapter:** `aiodoo-planner`  
**Dataset:** `datasets/v1.0.0/execution_dataset.jsonl` (5459 records)

## Invoke

```bash
python train.py --config configs/training/execution/experiment.yaml
```

Colab:

```python
TRAINING_ID = "execution"
experiment = ExperimentStore(workspace=workspace).load(TRAINING_ID)
```

## Prerequisites

1. Prior product adapter published at `models/adapters/aiodoo-planner/`
2. Dataset release `v1.0.0` present under workspace `datasets/v1.0.0/`
3. Base model `Qwen/Qwen3-8B` available via ModelStore / `AIODOO_COLAB_MODEL_PATH`

## Drive outputs

| Role | Path |
|------|------|
| Checkpoints | `training/cache/execution/checkpoints/` |
| Published adapter | `models/adapters/aiodoo-execution/` |
| Exports | `models/exports/aiodoo-execution/` |
| Run metadata | `experiments/execution/` |
