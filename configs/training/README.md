# Production Training Configurations

Public **capability** training IDs under `configs/training/<id>/`.

See [Capability Model](../../docs/capability_model.md),
[Product Model](../../docs/product_model.md), and
[Terminology](../../docs/terminology.md).

## Independent capability adapters

Each pack trains **one** capability adapter from the **base model** (fresh QLoRA).
There is **no** cross-capability `resume_from` chain.

`checkpointing.resume_from` is only for **same-run** recovery after an interrupt
(e.g. `training/cache/repair/checkpoints/checkpoint-200`).

**Product** packaging (Development / Reasoning) belongs in **`aiodoo-model`**,
not here. Training creates capabilities only.

| Catalog # | Capability ID | Dataset file | Records | Capability Package dir |
|----------:|---------------|----------------|--------:|------------------------|
| 1 | `coding` | `coding_v1_0.jsonl` | 5459 | `aiodoo-coding` |
| 2 | `planner` | `planner_v1_0.jsonl` | 5695 | `aiodoo-planner` |
| 3 | `execution` | `execution_dataset.jsonl` | 5459 | `aiodoo-execution` |
| 4 | `repair` | `repair_v1_0.jsonl` | 481 | `aiodoo-repair` |
| 5 | `context` | `context_v1_0.jsonl` | 50161 | `aiodoo-context` |
| 6 | `conversation` | `conversation_dataset.jsonl` | 1 | `aiodoo-conversation` |
| 7 | `approval` | `approval_dataset.jsonl` | 1 | `aiodoo-approval` |
| 8 | `evaluation` | `evaluation_dataset.jsonl` | 1 | `aiodoo-evaluation` |

Dataset root (Drive / local workspace):

```text
AIODOO/datasets/v1.0.0/
```

## Colab

```python
TRAINING_ID = "repair"  # capability id
experiment = ExperimentStore(workspace=workspace).load(TRAINING_ID)
```

Capability Package path:

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
