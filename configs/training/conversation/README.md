# conversation — Production Training Configuration

**Training ID:** `conversation`  
**Published adapter:** `aiodoo-conversation`  
**Catalog stage:** 6 / 8 (label only — not a weight chain)  
**Input adapter:** none (fresh from base model)  
**Dataset:** `datasets/v1.0.0/conversation_dataset.jsonl` (29,016 records)

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

1. Dataset release `v1.0.0` present under workspace `datasets/v1.0.0/`
2. Base model `Qwen/Qwen3-8B` available via ModelStore / `AIODOO_COLAB_MODEL_PATH`
3. After a Colab interrupt only: set `checkpointing.resume_from` to the last
   same-run checkpoint under `training/cache/conversation/checkpoints/`

## Drive outputs

| Role | Path |
|------|------|
| Checkpoints | `training/cache/conversation/checkpoints/` |
| Published adapter | `models/adapters/aiodoo-conversation/` |
| Exports | `models/exports/aiodoo-conversation/` |
| Run metadata | `experiments/conversation/` |
