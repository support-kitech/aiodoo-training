# coding — Production Training Configuration

**Training ID:** `coding`  
**Published adapter:** `aiodoo-coding`  
**Internal id (bookkeeping only):** `EXP-0001`

## Invoke

```bash
python train.py --config configs/training/coding/experiment.yaml
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
