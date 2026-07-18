"""Production publish contract — adapter inference files and validation handoff."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from aiodoo_training.artifacts.io_utils import ensure_parent_dir
from aiodoo_training.exceptions import ConfigError

# Checkpoint sidecars written by CheckpointManager — never published.
CHECKPOINT_SIDECAR_FILENAMES: frozenset[str] = frozenset(
    {
        "manifest.json",
        "rng.json",
        "dataset_session.json",
        "metrics.json",
        "checkpoints.json",
        "trainer_state.pt",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        "rng_state.pth",
    }
)

# Weight files required for a publishable PEFT adapter checkpoint.
ADAPTER_WEIGHT_FILENAMES: frozenset[str] = frozenset(
    {
        "adapter_model.safetensors",
        "adapter_model.bin",
    }
)

# Always required in a published adapter directory.
ADAPTER_REQUIRED_FILENAMES: frozenset[str] = frozenset({"adapter_config.json"})

# Optional tokenizer artifacts copied when present in the checkpoint.
TOKENIZER_FILENAMES: frozenset[str] = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    }
)

ARTIFACT_METADATA_FILENAME = "artifact.json"
PUBLISH_MANIFEST_FILENAME = "manifest.json"
VALIDATION_PROTOCOL_MAJOR = 1

_DATASET_TYPE_TO_ARTIFACT_TYPE: dict[str, str] = {
    "coding": "coding_adapter",
}


class PublishError(ConfigError):
    """Raised when adapter publish preflight or atomic write fails."""


def is_inference_artifact(filename: str) -> bool:
    """Return whether ``filename`` is allowed in a published adapter directory."""
    if filename in CHECKPOINT_SIDECAR_FILENAMES:
        return False
    if filename in ADAPTER_REQUIRED_FILENAMES or filename in ADAPTER_WEIGHT_FILENAMES:
        return True
    if filename in TOKENIZER_FILENAMES:
        return True
    if filename.startswith("adapter_model."):
        return True
    return False


def iter_inference_files(checkpoint_dir: Path) -> list[Path]:
    """List publishable files under a checkpoint directory (files only, non-recursive)."""
    if not checkpoint_dir.is_dir():
        return []
    return sorted(
        path
        for path in checkpoint_dir.iterdir()
        if path.is_file() and is_inference_artifact(path.name)
    )


def validate_checkpoint_for_publish(checkpoint_dir: Path) -> None:
    """Raise :class:`PublishError` when checkpoint cannot be published."""
    if not checkpoint_dir.is_dir():
        raise PublishError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    missing = [name for name in ADAPTER_REQUIRED_FILENAMES if not (checkpoint_dir / name).is_file()]
    if missing:
        raise PublishError(
            f"Checkpoint {checkpoint_dir} missing required adapter files: {sorted(missing)}"
        )

    if not any((checkpoint_dir / name).is_file() for name in ADAPTER_WEIGHT_FILENAMES):
        raise PublishError(
            f"Checkpoint {checkpoint_dir} missing adapter weights "
            f"(expected one of {sorted(ADAPTER_WEIGHT_FILENAMES)})"
        )


def verify_published_adapter(adapter_dir: Path) -> None:
    """Raise :class:`PublishError` when a published adapter directory is incomplete."""
    validate_checkpoint_for_publish(adapter_dir)
    artifact_path = adapter_dir / ARTIFACT_METADATA_FILENAME
    if not artifact_path.is_file():
        raise PublishError(f"Published adapter missing {ARTIFACT_METADATA_FILENAME}: {adapter_dir}")


def infer_adapter_artifact_type(resolved: dict[str, Any] | None) -> str:
    """Map resolved config dataset metadata to aiodoo-validation artifact_type."""
    if not isinstance(resolved, dict):
        return "coding_adapter"
    datasets = resolved.get("datasets")
    if isinstance(datasets, list):
        for entry in datasets:
            if isinstance(entry, dict):
                dtype = entry.get("dataset_type")
                if isinstance(dtype, str) and dtype in _DATASET_TYPE_TO_ARTIFACT_TYPE:
                    return _DATASET_TYPE_TO_ARTIFACT_TYPE[dtype]
    progressive = resolved.get("progressive_training") or resolved.get("skill_training")
    if isinstance(progressive, dict):
        stages = progressive.get("stages")
        if isinstance(stages, list) and stages:
            first = stages[0]
            if isinstance(first, dict):
                dtype = first.get("dataset_type") or first.get("name")
                if isinstance(dtype, str) and dtype in _DATASET_TYPE_TO_ARTIFACT_TYPE:
                    return _DATASET_TYPE_TO_ARTIFACT_TYPE[dtype]
    return "coding_adapter"


def build_adapter_artifact_json(
    *,
    experiment_id: str,
    resolved: dict[str, Any] | None = None,
    source_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Build ``artifact.json`` for aiodoo-validation adapter resolution."""
    artifact_type = infer_adapter_artifact_type(resolved)
    payload: dict[str, Any] = {
        "artifact_type": artifact_type,
        "protocol_major": VALIDATION_PROTOCOL_MAJOR,
        "identifier": experiment_id,
    }
    if artifact_type == "coding_adapter":
        adapter_type = "coding"
        if isinstance(resolved, dict):
            datasets = resolved.get("datasets")
            if isinstance(datasets, list) and datasets:
                first = datasets[0]
                if isinstance(first, dict) and isinstance(first.get("dataset_type"), str):
                    adapter_type = first["dataset_type"]
        payload["adapter_type"] = adapter_type
    if source_checkpoint:
        payload["source_checkpoint"] = source_checkpoint
    return payload


def build_base_model_artifact_json(
    *,
    model_id: str,
    model_family: str | None = None,
) -> dict[str, Any]:
    """Build ``artifact.json`` for aiodoo-validation base model resolution."""
    identifier = model_id.split("/")[-1].lower() if "/" in model_id else model_id.lower()
    family = model_family or identifier.split("-")[0]
    return {
        "artifact_type": "base_model",
        "protocol_major": VALIDATION_PROTOCOL_MAJOR,
        "identifier": identifier,
        "model_family": family,
        "model_id": model_id,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def atomic_replace_directory(tmp_dir: Path, dest_dir: Path) -> None:
    """
    Atomically promote ``tmp_dir`` to ``dest_dir``.

  Uses a backup rename so a failed promotion can restore the previous tree.
    """
    if not tmp_dir.is_dir():
        raise PublishError(f"Temporary publish directory missing: {tmp_dir}")

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir: Path | None = None
    if dest_dir.exists():
        backup_dir = dest_dir.parent / f".backup-{dest_dir.name}-{uuid.uuid4().hex[:8]}"
        dest_dir.rename(backup_dir)

    try:
        tmp_dir.rename(dest_dir)
    except OSError as exc:
        if backup_dir is not None and backup_dir.exists() and not dest_dir.exists():
            backup_dir.rename(dest_dir)
        raise PublishError(f"Failed to promote published adapter to {dest_dir}: {exc}") from exc

    if backup_dir is not None and backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)


def publish_inference_tree(
    *,
    source_files: list[Path],
    dest_dir: Path,
    metadata_files: dict[str, dict[str, Any]],
) -> Path:
    """
    Copy inference artifacts into a temp directory, verify, then atomically promote.

    ``metadata_files`` maps filename → JSON payload (e.g. artifact.json, manifest.json).
    """
    tmp_dir = dest_dir.parent / f".tmp-publish-{uuid.uuid4().hex[:12]}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    try:
        for src in source_files:
            shutil.copy2(src, tmp_dir / src.name)
        for filename, payload in metadata_files.items():
            write_json(tmp_dir / filename, payload)
        verify_published_adapter(tmp_dir)
        atomic_replace_directory(tmp_dir, dest_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return dest_dir


__all__ = [
    "ARTIFACT_METADATA_FILENAME",
    "CHECKPOINT_SIDECAR_FILENAMES",
    "PUBLISH_MANIFEST_FILENAME",
    "PublishError",
    "atomic_replace_directory",
    "build_adapter_artifact_json",
    "build_base_model_artifact_json",
    "infer_adapter_artifact_type",
    "is_inference_artifact",
    "iter_inference_files",
    "publish_inference_tree",
    "validate_checkpoint_for_publish",
    "verify_published_adapter",
    "write_json",
]
