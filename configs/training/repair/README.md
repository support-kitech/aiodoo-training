# repair — Production Training Configuration

**Training ID:** `repair`  
**Published adapter:** `aiodoo-repair`  
**Catalog stage:** 4 / 8 (label only — not a weight chain)  
**Input adapter:** none (fresh from base model)  
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

1. Dataset release `v1.0.0` present under workspace `datasets/v1.0.0/`
2. Base model `Qwen/Qwen3-8B` available via ModelStore / `AIODOO_COLAB_MODEL_PATH`
3. After a Colab interrupt only: set `checkpointing.resume_from` to the last
   same-run checkpoint under `training/cache/repair/checkpoints/`

## Drive outputs

| Role | Path |
|------|------|
| Checkpoints | `training/cache/repair/checkpoints/` |
| Published adapter | `models/adapters/aiodoo-repair/` |
| Exports | `models/exports/aiodoo-repair/` |
| Run metadata | `experiments/repair/` |
