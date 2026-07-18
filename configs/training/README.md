# Production Training Configurations

Public training IDs under `configs/training/<id>/`.

## Progressive curriculum (v1.0.0)

Train **one product at a time**. Each stage resumes from the prior published adapter.

| Stage | Training ID | Dataset file | Records | Input adapter | Output adapter |
|------:|-------------|--------------|--------:|---------------|----------------|
| 1 | `coding` | `coding_v1_0.jsonl` | 5459 | — | `aiodoo-coding` |
| 2 | `planner` | `planner_v1_0.jsonl` | 5695 | `aiodoo-coding` | `aiodoo-planner` |
| 3 | `execution` | `execution_dataset.jsonl` | 5459 | `aiodoo-planner` | `aiodoo-execution` |
| 4 | `repair` | `repair_v1_0.jsonl` | 481 | `aiodoo-execution` | `aiodoo-repair` |
| 5 | `context` | `context_v1_0.jsonl` | 50161 | `aiodoo-repair` | `aiodoo-context` |
| 6 | `conversation` | `conversation_dataset.jsonl` | 1 | `aiodoo-context` | `aiodoo-conversation` |
| 7 | `approval` | `approval_dataset.jsonl` | 1 | `aiodoo-conversation` | `aiodoo-approval` |
| 8 | `evaluation` | `evaluation_dataset.jsonl` | 1 | `aiodoo-approval` | `aiodoo-evaluation` |

Dataset root (Drive / local workspace):

```text
AIODOO/datasets/v1.0.0/
```

## Colab

```python
TRAINING_ID = "repair"  # or planner / execution / ...
experiment = ExperimentStore(workspace=workspace).load(TRAINING_ID)
```

Adapter product path:

```python
adapter_id = f"aiodoo-{TRAINING_ID}"  # e.g. aiodoo-repair
```

## Required fragments

Each training id directory must contain:

- `dataset.yaml`
- `model.yaml`
- `training.yaml`
- `evaluation.yaml`
- `export.yaml`
- `experiment.yaml` (train.py root include)
- `README.md`
