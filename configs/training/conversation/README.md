# conversation — Production Training Configuration

> **Dataset note:** v1.0.0 currently ships 1 conversation record — expand in a future dataset release.
**Training ID:** `conversation`  
**Published adapter:** `aiodoo-conversation`  
**Progressive stage:** 6 / 8  
**Input adapter:** `aiodoo-context`  
**Dataset:** `datasets/v1.0.0/conversation_dataset.jsonl` (1 records)

## Invoke

```bash
python train.py --config configs/training/conversation/experiment.yaml
```

Colab:

```python
TRAINING_ID = "conversation"
experiment = ExperimentStore(workspace=workspace).load(TRAINING_ID)
```

## Prerequisites

1. Prior product adapter published at `models/adapters/aiodoo-context/`
2. Dataset release `v1.0.0` present under workspace `datasets/v1.0.0/`
3. Base model `Qwen/Qwen3-8B` available via ModelStore / `AIODOO_COLAB_MODEL_PATH`

## Drive outputs

| Role | Path |
|------|------|
| Checkpoints | `training/cache/conversation/checkpoints/` |
| Published adapter | `models/adapters/aiodoo-conversation/` |
| Exports | `models/exports/aiodoo-conversation/` |
| Run metadata | `experiments/conversation/` |
