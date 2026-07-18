# approval — Production Training Configuration

> **Dataset note:** v1.0.0 currently ships 1 approval record — expand in a future dataset release.
**Training ID:** `approval`  
**Published adapter:** `aiodoo-approval`  
**Catalog stage:** 7 / 8 (label only — not a weight chain)  
**Input adapter:** none (fresh from base model)  
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

1. Dataset release `v1.0.0` present under workspace `datasets/v1.0.0/`
2. Base model `Qwen/Qwen3-8B` available via ModelStore / `AIODOO_COLAB_MODEL_PATH`
3. After a Colab interrupt only: set `checkpointing.resume_from` to the last
   same-run checkpoint under `training/cache/approval/checkpoints/`

## Drive outputs

| Role | Path |
|------|------|
| Checkpoints | `training/cache/approval/checkpoints/` |
| Published adapter | `models/adapters/aiodoo-approval/` |
| Exports | `models/exports/aiodoo-approval/` |
| Run metadata | `experiments/approval/` |
