# Artifact Output Pipeline

**Status:** Production artifact routing (Drive workspace `drive_v1`)

This document describes where every generated artifact lives in the canonical
AIODOO Drive workspace. The repository contains **source code only** — all
training outputs are routed through `ArtifactOutputManager`.

## Workspace root (required for production)

Production experiment configs declare `workspace.layout: drive_v1`. When active,
**`AIODOO_WORKSPACE_ROOT` must be set** to the AIODOO Drive workspace root.
Training fails fast if the variable is missing — the repository clone must
never be used as runtime storage.

```bash
export AIODOO_WORKSPACE_ROOT=/path/to/AIODOO
```

`aiodoo-colab` sets this variable explicitly before invoking `train.py`.
Training consumes it; workspace root is never inferred from Colab path hints.

## Canonical Drive layout

```text
AIODOO/
├── datasets/
├── training/
│   ├── aiodoo-training/                # source clone (not artifact storage)
│   └── cache/
│       └── coding/
│           └── checkpoints/            # transient training checkpoints
├── models/
│   ├── adapters/
│   │   └── aiodoo-coding/
│   │       ├── adapter_config.json
│   │       ├── adapter_model.safetensors
│   │       ├── tokenizer.json          # when present in checkpoint
│   │       ├── artifact.json           # validation handoff (required)
│   │       └── manifest.json           # training publish metadata
│   ├── merged/
│   │   └── aiodoo-coding/
│   │       ├── …                         # inference weights
│   │       ├── artifact.json
│   │       └── manifest.json
│   └── exports/
│       └── aiodoo-coding/
│           ├── bundle-coding-<fingerprint>/
│           │   ├── export_manifest.json
│           │   ├── checksums.txt
│           │   └── artifacts/
│           └── manifest.json
├── experiments/
│   └── coding/
│       ├── config/                     # config snapshot
│       ├── metrics/                      # history.jsonl
│       ├── validation/                   # evaluation reports
│       ├── logs/
│       │   └── tracking/                 # run tracking JSONL
│       └── summary.json
└── cache/                                # optional workspace cache
```

Base Hugging Face models are cached on Colab local SSD
(`/content/aiodoo-model-cache/<org>__<name>/`), not on Drive. Training writes
`artifact.json` there during finalize for validation handoff.

## Artifact destinations

| Artifact | Destination | Written by |
|----------|-------------|------------|
| Training checkpoints | `training/cache/{training_id}/checkpoints/` | `CheckpointManager` |
| Final adapter | `models/adapters/{aiodoo-<training_id>}/` | `ArtifactOutputManager.publish_adapter_from_checkpoint` |
| Merged model | `models/merged/{aiodoo-<training_id>}/` | `ArtifactOutputManager.publish_merged_from_bundle` |
| Export bundles | `models/exports/{aiodoo-<training_id>}/` | `ExportManager` |
| Metrics history | `experiments/{training_id}/metrics/history.jsonl` | `TrainingHistory` |
| Validation reports | `experiments/{training_id}/validation/` | Evaluation stage |
| Run tracking | `experiments/{training_id}/logs/tracking/` | `FilesystemTracker` |
| Experiment summary | `experiments/{training_id}/summary.json` | `ArtifactOutputManager.write_experiment_summary` |
| Config snapshot | `experiments/{training_id}/config/` | `ArtifactOutputManager.snapshot_config` |
| Base model `artifact.json` | Colab model cache dir | `ArtifactOutputManager.publish_base_model_artifact` |

## Publish rules

1. **Inference-only adapters** — published adapters exclude checkpoint sidecars
   (`rng.json`, `metrics.json`, `dataset_session.json`, trainer state).
2. **Atomic publish** — write to `.tmp-publish-*`, verify, then rename.
3. **Validation handoff** — every published adapter includes `artifact.json`
   consumed directly by `aiodoo-validation` (no manual bridge files).
4. **One destination per artifact** — no duplicate copies across repo and Drive.
5. **No empty folders** — directories are created only immediately before writing a file.
6. **Repository is not storage** — `artifacts/` in the repo is for local dev/CI only (gitignored).

## Metadata ownership

| File | Owner | Consumer |
|------|-------|----------|
| `artifact.json` | Training publish | `aiodoo-validation` artifact resolution |
| `manifest.json` (adapter dir) | Training publish | Training diagnostics |
| `export_manifest.json` | Export bundle | `aiodoo-models` (future) |
| `checksums.txt` | Export bundle | `aiodoo-models` integrity |

## Cleanup utility

Remove empty cache directories, stale `.tmp-*` trees, and abandoned checkpoints:

```bash
# Dry-run (default)
python cleanup_artifacts.py /path/to/AIODOO

# Delete mode
python cleanup_artifacts.py /path/to/AIODOO --delete

# Or via environment
export AIODOO_WORKSPACE_ROOT=/path/to/AIODOO
python cleanup_artifacts.py --delete
```

Protected trees (never deleted): `models/adapters/`, `models/merged/`,
`models/exports/`, `experiments/`.

## Integration with aiodoo-validation

Point validation at:

- `--base-model` → Colab SSD cache dir (with `artifact.json`)
- `--adapter` → `models/adapters/{aiodoo-<training_id>}/` (with `artifact.json`)

No changes to `aiodoo-validation` are required.

See also: [SMOKE.md](SMOKE.md) for the end-to-end smoke procedure and
[Artifact Contract](artifact_contract.md) for the Training → Models handoff.
