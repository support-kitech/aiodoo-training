# execution — Production Training Configuration

**Training ID:** `execution`  
**Published adapter:** `aiodoo-execution`  
**Catalog stage:** 3 / 8 (label only — not a weight chain)  
**Input adapter:** none (fresh from base model)  
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

1. Dataset release `v1.0.0` present under workspace `datasets/v1.0.0/`
2. Base model `Qwen/Qwen3-8B` available via ModelStore / `AIODOO_COLAB_MODEL_PATH`
3. After a Colab interrupt only: set `checkpointing.resume_from` to the last
   same-run checkpoint under `training/cache/execution/checkpoints/`

## Drive outputs

| Role | Path |
|------|------|
| Checkpoints | `training/cache/execution/checkpoints/` |
| Published adapter | `models/adapters/aiodoo-execution/` |
| Exports | `models/exports/aiodoo-execution/` |
| Run metadata | `experiments/execution/` |
