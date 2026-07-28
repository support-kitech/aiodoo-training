# evaluation — Production Training Configuration

**Training ID:** `evaluation`  
**Published adapter:** `aiodoo-evaluation`  
**Catalog stage:** 8 / 8 (label only — not a weight chain)  
**Input adapter:** none (fresh from base model)  
**Dataset:** `datasets/v1.0.0/evaluation_dataset.jsonl` (**189,615** judgment SFT records)

Contract path: `project_evaluation` → `CapabilityPromptBuilder` →
system/user/assistant (`EvaluationRequest` / `EvaluationResponse`).

Do **not** train on `evaluation_benchmark_catalog.jsonl` (certification only).

## Invoke

```bash
python train.py --config configs/training/evaluation/experiment.yaml
```

Colab:

```python
TRAINING_ID = "evaluation"
experiment = ExperimentStore(workspace=workspace).load(TRAINING_ID)
```

## Prerequisites

1. Dataset release `v1.0.0` present under workspace `datasets/v1.0.0/`
   with certified `evaluation_dataset.jsonl` (judgment SFT)
2. Base model `Qwen/Qwen3-8B` available via ModelStore / `AIODOO_COLAB_MODEL_PATH`
3. After a Colab interrupt only: set `checkpointing.resume_from` to the last
   same-run checkpoint under `training/cache/evaluation/checkpoints/`

## Drive outputs

| Role | Path |
|------|------|
| Checkpoints | `training/cache/evaluation/checkpoints/` |
| Published adapter | `models/adapters/aiodoo-evaluation/` |
| Exports | `models/exports/aiodoo-evaluation/` |
| Run metadata | `experiments/evaluation/` |
