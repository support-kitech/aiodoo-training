# coding — Production Training Configuration

**Training ID:** `coding`  
**Published adapter:** `aiodoo-coding`  
**Progressive stage:** 1 / 8  
**Input adapter:** none (base model)  
**Dataset:** `datasets/v1.0.0/coding_v1_0.jsonl` (5459 records)  
**Internal id (bookkeeping only):** `EXP-0001`

## Invoke

```bash
python train.py --config configs/training/coding/experiment.yaml
```

Colab:

```python
TRAINING_ID = "coding"
experiment = ExperimentStore(workspace=workspace).load(TRAINING_ID)
```

## Layout

```text
configs/training/coding/
  experiment.yaml   # root + identity
  dataset.yaml
  model.yaml
  training.yaml
  evaluation.yaml
  export.yaml
```

See [`../README.md`](../README.md) for the full progressive curriculum.

## Drive outputs (canonical)

| Role | Path |
|------|------|
| Checkpoints | `training/cache/coding/checkpoints/` |
| Published adapter | `models/adapters/aiodoo-coding/` |
| Merged | `models/merged/aiodoo-coding/` |
| Exports | `models/exports/aiodoo-coding/` |
| Run metadata | `experiments/coding/` |

Legacy `EXP-0001` folder names are internal metadata only and must never appear
in Drive paths, adapter product names, or notebook UI.
