# Production Smoke Test — coding

End-to-end smoke procedure for the **coding** production training product.
All paths follow `aiodoo-training` `ArtifactOutputLayout` — the single layout
authority for Drive outputs.

## Drive layout

```text
AIODOO/
├── training/
│   └── cache/
│       └── coding/
│           └── checkpoints/
├── models/
│   ├── adapters/
│   │   └── aiodoo-coding/                   # published adapter + artifact.json
│   ├── merged/
│   │   └── aiodoo-coding/                   # optional (when export merges)
│   └── exports/
│       └── aiodoo-coding/
│           └── bundle-coding-*/             # export bundle
├── experiments/
│   └── coding/
│       ├── summary.json
│       ├── config/
│       ├── metrics/
│       └── logs/
└── datasets/
```

## Colab / notebook

```python
TRAINING_ID = "coding"
experiment = ExperimentStore(workspace=ws).load(TRAINING_ID)
```

Legacy `EXP-0001` is accepted by loaders and normalizes to `coding`, but must
never appear in Drive folders, adapter product names, or notebook UI labels.

## CLI

```bash
export AIODOO_WORKSPACE_ROOT=/path/to/AIODOO
python train.py --config configs/training/coding/experiment.yaml
```

## Validation handoff

```bash
aiodoo-validate \
  --adapter "$AIODOO_WORKSPACE_ROOT/models/adapters/aiodoo-coding" \
  --odoo-versions 18
```

## Checklist

1. `ArtifactOutputLayout` path contract
2. Checkpoints under `training/cache/coding/checkpoints/`
3. Published adapter under `models/adapters/aiodoo-coding/`
4. No `EXP-*` directories under Drive
5. Notebook shows `Training : coding`
