"""Production publish contract — Capability Package files and validation handoff.

Capability Package ``artifact.json`` field roles (Option A / ADR-0022):

* ``artifact_type`` — frozen protocol/package kind (``coding_adapter``,
  ``base_model``, ``merged_model``). Never invent per-capability type strings.
* ``capability_id`` — business skill identity (``coding``, ``repair``, …).
* ``adapter_type`` — skill label required by frozen ``aiodoo-validation``
  profile matching (must equal the validation profile / capability id).
  Retained for compatibility; not the PEFT implementation kind.
* ``peft_type`` — adaptation implementation (``lora``, ``qlora``, …) for
  ``aiodoo-model`` normalize (registry ``adapter_type`` becomes PEFT kind).
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiodoo_training.artifacts.io_utils import ensure_parent_dir
from aiodoo_training.exceptions import ConfigError
from aiodoo_training.naming import (
    TRAINING_IDS,
    adapter_product_id,
    is_training_id,
    normalize_training_id,
    resolve_public_training_id,
)

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

# Frozen aiodoo-validation / aiodoo-model protocol kind for PEFT Capability Packages.
ADAPTER_PROTOCOL_ARTIFACT_TYPE = "coding_adapter"
BASE_PROTOCOL_ARTIFACT_TYPE = "base_model"
MERGED_PROTOCOL_ARTIFACT_TYPE = "merged_model"

# Catalog capability → protocol artifact_type (all PEFT adapters share coding_adapter).
_CAPABILITY_TO_PROTOCOL_ARTIFACT_TYPE: dict[str, str] = {
    capability_id: ADAPTER_PROTOCOL_ARTIFACT_TYPE for capability_id in TRAINING_IDS
}

# Default Odoo versions when config does not declare supported_odoo_versions.
DEFAULT_SUPPORTED_ODOO_VERSIONS: tuple[int, ...] = (17, 18, 19)

_PRODUCER_NAME = "aiodoo-training"


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


def resolve_capability_id(resolved: dict[str, Any] | None) -> str:
    """
    Resolve the capability id for a Capability Package.

    Preference: public training id from experiment config, then dataset_type /
    skill_training stage name, else ``coding`` only as last resort when no
    catalog signal exists (legacy / incomplete configs).
    """
    if isinstance(resolved, dict):
        try:
            return resolve_public_training_id(resolved)
        except ValueError:
            pass
        for candidate in _iter_capability_candidates(resolved):
            if is_training_id(candidate):
                return candidate
            try:
                return normalize_training_id(candidate)
            except ValueError:
                continue
    return "coding"


def infer_adapter_artifact_type(resolved: dict[str, Any] | None) -> str:
    """
    Return the frozen protocol ``artifact_type`` for a PEFT Capability Package.

    Always ``coding_adapter`` for catalog capabilities — business identity is
    ``capability_id``, not a distinct artifact_type string.
    """
    capability_id = resolve_capability_id(resolved)
    return _CAPABILITY_TO_PROTOCOL_ARTIFACT_TYPE.get(
        capability_id, ADAPTER_PROTOCOL_ARTIFACT_TYPE
    )


def build_adapter_artifact_json(
    *,
    experiment_id: str,
    resolved: dict[str, Any] | None = None,
    source_checkpoint: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build Capability Package ``artifact.json`` for adapter Drive publish."""
    capability_id = resolve_capability_id(resolved)
    identifier = experiment_id.strip() if experiment_id.strip() else adapter_product_id(capability_id)
    family, architecture = _resolve_family_architecture(resolved)
    payload: dict[str, Any] = {
        "artifact_type": ADAPTER_PROTOCOL_ARTIFACT_TYPE,
        "protocol_major": VALIDATION_PROTOCOL_MAJOR,
        "identifier": identifier,
        "capability_id": capability_id,
        # Frozen validation profiles require adapter_type == profile/capability id.
        "adapter_type": capability_id,
        "peft_type": _resolve_peft_type(resolved),
        "supported_odoo_versions": list(_resolve_supported_odoo_versions(resolved)),
        "created_at": created_at or _utc_now_iso(),
        "producer": _PRODUCER_NAME,
        "training_version": _training_version(),
        "training_source": capability_id,
    }
    if family is not None:
        payload["model_family"] = family
    if architecture is not None:
        payload["architecture"] = architecture
    dataset_version = _resolve_dataset_version(resolved)
    if dataset_version is not None:
        payload["dataset_version"] = dataset_version
    if source_checkpoint:
        payload["source_checkpoint"] = source_checkpoint
    return payload


def build_merged_artifact_json(
    *,
    experiment_id: str,
    resolved: dict[str, Any] | None = None,
    source_bundle: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build Capability Package ``artifact.json`` for merged Drive publish."""
    capability_id = resolve_capability_id(resolved)
    identifier = experiment_id.strip() if experiment_id.strip() else adapter_product_id(capability_id)
    family, architecture = _resolve_family_architecture(resolved)
    payload: dict[str, Any] = {
        "artifact_type": MERGED_PROTOCOL_ARTIFACT_TYPE,
        "protocol_major": VALIDATION_PROTOCOL_MAJOR,
        "identifier": identifier,
        "capability_id": capability_id,
        "adapter_type": capability_id,
        "supported_odoo_versions": list(_resolve_supported_odoo_versions(resolved)),
        "created_at": created_at or _utc_now_iso(),
        "producer": _PRODUCER_NAME,
        "training_version": _training_version(),
        "training_source": capability_id,
    }
    if family is not None:
        payload["model_family"] = family
    if architecture is not None:
        payload["architecture"] = architecture
    dataset_version = _resolve_dataset_version(resolved)
    if dataset_version is not None:
        payload["dataset_version"] = dataset_version
    if source_bundle:
        payload["source_bundle"] = source_bundle
    return payload


def build_base_model_artifact_json(
    *,
    model_id: str,
    model_family: str | None = None,
    architecture: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build ``artifact.json`` for aiodoo-validation base model resolution."""
    identifier = model_id.split("/")[-1].lower() if "/" in model_id else model_id.lower()
    family = model_family or identifier.split("-")[0]
    arch = architecture or family
    return {
        "artifact_type": BASE_PROTOCOL_ARTIFACT_TYPE,
        "protocol_major": VALIDATION_PROTOCOL_MAJOR,
        "identifier": identifier,
        "model_family": family,
        "architecture": arch,
        "model_id": model_id,
        "created_at": created_at or _utc_now_iso(),
        "producer": _PRODUCER_NAME,
        "training_version": _training_version(),
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


def _iter_capability_candidates(resolved: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    datasets = resolved.get("datasets")
    if isinstance(datasets, list):
        for entry in datasets:
            if isinstance(entry, dict):
                dtype = entry.get("dataset_type")
                if isinstance(dtype, str) and dtype.strip():
                    candidates.append(dtype.strip())
    progressive = resolved.get("progressive_training") or resolved.get("skill_training")
    if isinstance(progressive, dict):
        stages = progressive.get("stages")
        if isinstance(stages, list):
            for stage in stages:
                if isinstance(stage, dict):
                    for key in ("dataset_type", "name"):
                        value = stage.get(key)
                        if isinstance(value, str) and value.strip():
                            candidates.append(value.strip())
    return candidates


def _resolve_peft_type(resolved: dict[str, Any] | None) -> str:
    if isinstance(resolved, dict):
        adaptation = resolved.get("adaptation")
        if isinstance(adaptation, dict):
            for key in ("adapter_type", "strategy", "peft_type"):
                value = adaptation.get(key)
                if isinstance(value, str) and value.strip():
                    text = value.strip().lower()
                    if text in {"lora", "qlora", "full"}:
                        return text
    return "lora"


def _resolve_family_architecture(
    resolved: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    if not isinstance(resolved, dict):
        return None, None
    model = resolved.get("model")
    if not isinstance(model, dict):
        return None, None
    family = model.get("family") or model.get("model_family")
    family_text = family.strip() if isinstance(family, str) and family.strip() else None
    architecture = model.get("architecture")
    architecture_text = (
        architecture.strip()
        if isinstance(architecture, str) and architecture.strip()
        else family_text
    )
    return family_text, architecture_text


def _resolve_supported_odoo_versions(resolved: dict[str, Any] | None) -> tuple[int, ...]:
    if isinstance(resolved, dict):
        for container in (
            resolved,
            resolved.get("metadata") if isinstance(resolved.get("metadata"), dict) else None,
            resolved.get("model") if isinstance(resolved.get("model"), dict) else None,
            resolved.get("export") if isinstance(resolved.get("export"), dict) else None,
        ):
            if not isinstance(container, dict):
                continue
            raw = container.get("supported_odoo_versions") or container.get("odoo_versions")
            parsed = _as_int_tuple(raw)
            if parsed:
                return parsed
    return DEFAULT_SUPPORTED_ODOO_VERSIONS


def _resolve_dataset_version(resolved: dict[str, Any] | None) -> str | None:
    if not isinstance(resolved, dict):
        return None
    value = resolved.get("dataset_version")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _as_int_tuple(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            return tuple(int(v) for v in value)
        except (TypeError, ValueError):
            return None
    return None


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _training_version() -> str:
    from aiodoo_training import __version__

    return str(__version__)


__all__ = [
    "ADAPTER_PROTOCOL_ARTIFACT_TYPE",
    "ARTIFACT_METADATA_FILENAME",
    "BASE_PROTOCOL_ARTIFACT_TYPE",
    "CHECKPOINT_SIDECAR_FILENAMES",
    "DEFAULT_SUPPORTED_ODOO_VERSIONS",
    "MERGED_PROTOCOL_ARTIFACT_TYPE",
    "PUBLISH_MANIFEST_FILENAME",
    "PublishError",
    "atomic_replace_directory",
    "build_adapter_artifact_json",
    "build_base_model_artifact_json",
    "build_merged_artifact_json",
    "infer_adapter_artifact_type",
    "is_inference_artifact",
    "iter_inference_files",
    "publish_inference_tree",
    "resolve_capability_id",
    "validate_checkpoint_for_publish",
    "verify_published_adapter",
    "write_json",
]
