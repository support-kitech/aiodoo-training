# context — Production Training Configuration

**Training ID:** `context`  
**Published adapter:** `aiodoo-context`  
**Progressive stage:** 5 / 8  
**Input adapter:** `aiodoo-repair`  
**Dataset:** `datasets/v1.0.0/context_v1_0.jsonl` (50161 records)

## Invoke

```bash
python train.py --config configs/training/context/experiment.yaml
```

Colab:

```python
TRAINING_ID = "context"
experiment = ExperimentStore(workspace=workspace).load(TRAINING_ID)
```

## Prerequisites

1. Prior product adapter published at `models/adapters/aiodoo-repair/`
2. Dataset release `v1.0.0` present under workspace `datasets/v1.0.0/`
3. Base model `Qwen/Qwen3-8B` available via ModelStore / `AIODOO_COLAB_MODEL_PATH`

## Drive outputs

| Role | Path |
|------|------|
| Checkpoints | `training/cache/context/checkpoints/` |
| Published adapter | `models/adapters/aiodoo-context/` |
| Exports | `models/exports/aiodoo-context/` |
| Run metadata | `experiments/context/` |
