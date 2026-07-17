# coding — Smoke Test Procedure

See also repository-level [docs/SMOKE.md](../../../docs/SMOKE.md).

1. Set `AIODOO_WORKSPACE_ROOT` to the AIODOO Drive workspace.
2. Launch with `TRAINING_ID = "coding"` (notebook) or:

```bash
python train.py --config configs/training/coding/experiment.yaml
```

3. Confirm checkpoints under `training/cache/coding/checkpoints/`.
4. Confirm published adapter under `models/adapters/aiodoo-coding/`.
5. Confirm no `EXP-*` directories appear under Drive.
