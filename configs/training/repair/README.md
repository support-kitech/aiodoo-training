# repair — Production Training Configuration

**Training ID:** `repair`  
**Published adapter:** `aiodoo-repair`  
**Progressive stage:** 4 / 8  
**Input adapter:** `aiodoo-execution`  
**Dataset:** `datasets/v1.0.0/repair_v1_0.jsonl` (481 records)

## Invoke

```bash
python train.py --config configs/training/repair/experiment.yaml
```

Colab:

```python
TRAINING_ID = "repair"
experiment = ExperimentStore(workspace=workspace).load(TRAINING_ID)
```

## Prerequisites

1. Prior product adapter published at `models/adapters/aiodoo-execution/`
2. Dataset release `v1.0.0` present under workspace `datasets/v1.0.0/`
3. Base model `Qwen/Qwen3-8B` available via ModelStore / `AIODOO_COLAB_MODEL_PATH`

## Drive outputs

| Role | Path |
|------|------|
| Checkpoints | `training/cache/repair/checkpoints/` |
| Published adapter | `models/adapters/aiodoo-repair/` |
| Exports | `models/exports/aiodoo-repair/` |
| Run metadata | `experiments/repair/` |
