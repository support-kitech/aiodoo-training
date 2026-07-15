# EXP-0001 Configuration Isolation

## Resolution chain

When executing:

```bash
python train.py --config configs/experiments/production/EXP-0001/experiment.yaml
```

the loader (`ConfigSystem.composer`) does **exactly**:

```text
experiment.yaml
├── include → dataset.yaml
├── include → model.yaml
├── include → training.yaml
├── include → evaluation.yaml
└── include → export.yaml
```

Then (in process, not from other YAML files):

```text
composed mapping
  → RawExperimentModel validation
  → path resolve (relative → absolute vs experiment directory)
  → to_experiment_config()  (typed ExperimentConfig + fragment parsers)
  → optional AIODOO_COLAB_* env overlays
  → Pipeline
```

**No other YAML under `configs/` is loaded.**

Specifically **not** loaded:

- `configs/experiments/example.yaml`
- `configs/training/default.yaml`
- `configs/adaptation/lora-r8.yaml`
- `configs/datasets/coding-example.yaml`
- `configs/models/qwen25-coder-0.5b.yaml`
- `configs/export/default.yaml`

## Self-containment

All production knobs for EXP-0001 live under this directory as sibling fragments
referenced only by `experiment.yaml` `include:`.

Parser / domain **defaults** (e.g. Pydantic field defaults when a key is absent)
still apply to *missing* keys. EXP-0001 explicitly sets every production-critical
key so those defaults are not relied upon for model id, QLoRA, T4 precision,
batching, packing, curriculum, checkpoints, eval, or export.

## Coexistence with `example.yaml`

| Concern | Status |
|---------|--------|
| Filename collision | None — different paths |
| Include collision | None — example pulls `../` templates; EXP-0001 pulls siblings only |
| Loader ambiguity | None — `--config` path is explicit |
| Colab preference | Prefers `configs/experiments/production/EXP-NNNN/experiment.yaml` |

`example.yaml` is for documentation, unit tests, and framework smoke only.
