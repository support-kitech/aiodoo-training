# planner — Production Training Configuration

**Training ID:** `planner`  
**Published adapter:** `aiodoo-planner`  
**Catalog stage:** 2 / 8 (label only — not a weight chain)  
**Input adapter:** none (fresh from base model)  
**Dataset:** `datasets/v1.0.0/planner_v1_0.jsonl` (5695 records)

## Invoke

```bash
python train.py --config configs/training/planner/experiment.yaml
```

Colab:

```python
TRAINING_ID = "planner"
experiment = ExperimentStore(workspace=workspace).load(TRAINING_ID)
```

## Prerequisites

1. Dataset release `v1.0.0` present under workspace `datasets/v1.0.0/`
2. Base model `Qwen/Qwen3-8B` available via ModelStore / `AIODOO_COLAB_MODEL_PATH`
3. After a Colab interrupt only: set `checkpointing.resume_from` to the last
   same-run checkpoint under `training/cache/planner/checkpoints/`

## Drive outputs

| Role | Path |
|------|------|
| Checkpoints | `training/cache/planner/checkpoints/` |
| Published adapter | `models/adapters/aiodoo-planner/` |
| Exports | `models/exports/aiodoo-planner/` |
| Run metadata | `experiments/planner/` |
