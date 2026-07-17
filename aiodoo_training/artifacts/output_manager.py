"""Central artifact output manager — routes all production writes to Drive."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiodoo_training.artifacts.io_utils import ensure_parent_dir
from aiodoo_training.artifacts.output_layout import ArtifactOutputLayout
from aiodoo_training.artifacts.publish_contract import (
    ARTIFACT_METADATA_FILENAME,
    PUBLISH_MANIFEST_FILENAME,
    PublishError,
    build_adapter_artifact_json,
    build_base_model_artifact_json,
    iter_inference_files,
    publish_inference_tree,
    validate_checkpoint_for_publish,
    write_json,
)
from aiodoo_training.exceptions import ConfigError

_WORKSPACE_ENV = "AIODOO_WORKSPACE_ROOT"
_DRIVE_LAYOUT = "drive_v1"


def resolve_workspace_root(resolved: dict[str, Any] | None = None) -> Path | None:
    """
    Resolve the AIODOO Drive workspace root.

    Priority:
    1. ``AIODOO_WORKSPACE_ROOT`` environment variable
    2. ``workspace.root`` in resolved config

    Colab path hints are never inferred — callers must set ``AIODOO_WORKSPACE_ROOT``.
    """
    raw = os.environ.get(_WORKSPACE_ENV)
    if raw and str(raw).strip():
        return Path(raw)

    if resolved:
        workspace = resolved.get("workspace")
        if isinstance(workspace, dict):
            root = workspace.get("root")
            if isinstance(root, str) and root.strip():
                return Path(root)

    return None


def requires_drive_workspace(resolved: dict[str, Any] | None) -> bool:
    """Return whether the resolved config opts into the production Drive layout."""
    workspace = (resolved or {}).get("workspace")
    if isinstance(workspace, dict) and workspace.get("layout") == _DRIVE_LAYOUT:
        return True
    return False


def validate_drive_workspace_contract(resolved: dict[str, Any]) -> None:
    """
    Fail fast when ``workspace.layout == drive_v1`` without a workspace root.

    Production runs must set ``AIODOO_WORKSPACE_ROOT`` (or ``workspace.root`` in
    resolved config). Repository-relative paths are never used as runtime storage.
    """
    if not requires_drive_workspace(resolved):
        return
    root = resolve_workspace_root(resolved)
    if root is None:
        raise ConfigError(
            "Production workspace contract violation: workspace.layout is "
            f"'{_DRIVE_LAYOUT}' but {_WORKSPACE_ENV} is not set. "
            "Set AIODOO_WORKSPACE_ROOT to the AIODOO Drive workspace root "
            "(e.g. /content/drive/MyDrive/colab_notebooks/AIODOO). "
            "The repository clone must never be used as runtime storage."
        )
    if not root.is_dir():
        raise ConfigError(
            f"Production workspace root does not exist or is not a directory: {root}"
        )


def should_use_canonical_layout(resolved: dict[str, Any] | None = None) -> bool:
    """Return whether outputs should be routed through the canonical Drive layout."""
    return resolve_workspace_root(resolved) is not None


def resolve_experiment_id(resolved: dict[str, Any]) -> str:
    experiment = resolved.get("experiment")
    if isinstance(experiment, dict) and isinstance(experiment.get("id"), str):
        return experiment["id"]
    name = resolved.get("name")
    if isinstance(name, str) and name.strip():
        return name
    return "unknown"


@dataclass(frozen=True, slots=True)
class ArtifactOutputManager:
    """Routes every generated artifact to exactly one canonical destination."""

    layout: ArtifactOutputLayout
    resolved: dict[str, Any] | None = None

    @classmethod
    def from_resolved(cls, resolved: dict[str, Any]) -> ArtifactOutputManager | None:
        """Build manager when canonical layout is active; otherwise return None."""
        if not should_use_canonical_layout(resolved):
            return None
        root = resolve_workspace_root(resolved)
        if root is None:
            return None
        experiment_id = resolve_experiment_id(resolved)
        return cls(
            layout=ArtifactOutputLayout(workspace_root=root, experiment_id=experiment_id),
            resolved=resolved,
        )

    def apply_to_resolved(self, resolved: dict[str, Any]) -> dict[str, Any]:
        """Rewrite output paths in a resolved config mapping to canonical layout."""
        data = dict(resolved)
        layout = self.layout

        ckpt = dict(data.get("checkpointing") or {})
        ckpt["output_dir"] = str(layout.adapter_checkpoints_dir)
        data["checkpointing"] = ckpt

        export = dict(data.get("export") or {})
        export["output_dir"] = str(layout.export_dir)
        data["export"] = export

        metrics = dict(data.get("metrics") or {})
        metrics["history_path"] = str(layout.metrics_history_path)
        data["metrics"] = metrics

        tracking = dict(data.get("tracking") or {})
        tracking["root_dir"] = str(layout.tracking_root)
        data["tracking"] = tracking

        workspace = dict(data.get("workspace") or {})
        workspace["root"] = str(layout.workspace_root)
        workspace["layout"] = _DRIVE_LAYOUT
        workspace["experiment_id"] = layout.experiment_id
        data["workspace"] = workspace

        return data

    def write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        ensure_parent_dir(path)
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return path

    def copy_tree(self, source: Path, destination: Path) -> Path:
        """Copy a directory tree to a canonical destination (no empty dirs)."""
        if not source.exists():
            return destination
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        return destination

    def copy_file(self, source: Path, destination: Path) -> Path:
        if not source.is_file():
            return destination
        ensure_parent_dir(destination)
        shutil.copy2(source, destination)
        return destination

    def publish_adapter_from_checkpoint(self, checkpoint_dir: Path) -> Path | None:
        """
        Publish inference-only adapter artifacts from a checkpoint directory.

        Validates the checkpoint, copies only inference files, writes
        ``artifact.json`` for aiodoo-validation, and promotes atomically.
        """
        if not checkpoint_dir.is_dir():
            return None

        validate_checkpoint_for_publish(checkpoint_dir)
        source_files = iter_inference_files(checkpoint_dir)
        if not source_files:
            raise PublishError(f"No inference artifacts found in checkpoint: {checkpoint_dir}")

        published_at = datetime.now(UTC).isoformat()
        manifest = {
            "experiment_id": self.layout.experiment_id,
            "source_checkpoint": str(checkpoint_dir),
            "published_at": published_at,
            "artifact_type": "peft_adapter",
        }
        artifact_json = build_adapter_artifact_json(
            experiment_id=self.layout.experiment_id,
            resolved=self.resolved,
            source_checkpoint=str(checkpoint_dir),
        )

        dest = self.layout.adapter_dir
        publish_inference_tree(
            source_files=source_files,
            dest_dir=dest,
            metadata_files={
                PUBLISH_MANIFEST_FILENAME: manifest,
                ARTIFACT_METADATA_FILENAME: artifact_json,
            },
        )
        return dest

    def publish_base_model_artifact(self, model_dir: Path) -> Path | None:
        """Write ``artifact.json`` beside a base model directory for validation handoff."""
        if not model_dir.is_dir():
            return None
        model_id = _resolve_model_id(self.resolved, model_dir)
        model_family = _resolve_model_family(self.resolved)
        payload = build_base_model_artifact_json(model_id=model_id, model_family=model_family)
        write_json(model_dir / ARTIFACT_METADATA_FILENAME, payload)
        return model_dir / ARTIFACT_METADATA_FILENAME

    def publish_merged_from_bundle(self, bundle_root: Path) -> Path | None:
        """Copy merged weights from an export bundle to ``models/merged/{EXP}/``."""
        merged_src = bundle_root / "artifacts" / "merged"
        if not merged_src.is_dir():
            return None
        dest = self.layout.merged_dir
        tmp_dir = dest.parent / f".tmp-merged-{self.layout.experiment_id}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.copytree(merged_src, tmp_dir)
        manifest = {
            "experiment_id": self.layout.experiment_id,
            "source_bundle": str(bundle_root),
            "published_at": datetime.now(UTC).isoformat(),
            "artifact_type": "merged_model",
        }
        write_json(tmp_dir / PUBLISH_MANIFEST_FILENAME, manifest)
        merged_artifact = {
            "artifact_type": "merged_model",
            "protocol_major": 1,
            "identifier": self.layout.experiment_id,
        }
        write_json(tmp_dir / ARTIFACT_METADATA_FILENAME, merged_artifact)
        from aiodoo_training.artifacts.publish_contract import atomic_replace_directory

        atomic_replace_directory(tmp_dir, dest)
        return dest

    def write_experiment_summary(
        self,
        *,
        run_id: str,
        success: bool,
        duration_seconds: float,
        paths: dict[str, str | None],
        extra: dict[str, Any] | None = None,
    ) -> Path:
        payload: dict[str, Any] = {
            "experiment_id": self.layout.experiment_id,
            "run_id": run_id,
            "success": success,
            "duration_seconds": duration_seconds,
            "paths": paths,
            "published_at": datetime.now(UTC).isoformat(),
        }
        if extra:
            payload.update(extra)
        return self.write_json(self.layout.summary_path, payload)

    def snapshot_config(self, config_path: Path) -> Path | None:
        """Copy experiment config snapshot into ``experiments/{EXP}/config/``."""
        if not config_path.is_file():
            return None
        dest = self.layout.experiment_config_dir / config_path.name
        return self.copy_file(config_path, dest)

    def find_latest_checkpoint(self, checkpoints_dir: Path) -> Path | None:
        """Return the highest-step checkpoint directory under ``checkpoints_dir``."""
        if not checkpoints_dir.is_dir():
            return None
        candidates: list[tuple[int, Path]] = []
        for child in checkpoints_dir.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            step = _parse_checkpoint_step(child.name)
            if step is not None:
                candidates.append((step, child))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]


def _resolve_model_id(resolved: dict[str, Any] | None, model_dir: Path) -> str:
    if isinstance(resolved, dict):
        model = resolved.get("model")
        if isinstance(model, dict):
            for key in ("model_id", "base_model", "name", "id"):
                value = model.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    return model_dir.name.replace("__", "/")


def _resolve_model_family(resolved: dict[str, Any] | None) -> str | None:
    if isinstance(resolved, dict):
        model = resolved.get("model")
        if isinstance(model, dict):
            family = model.get("family") or model.get("model_family")
            if isinstance(family, str) and family.strip():
                return family
    return None


def _parse_checkpoint_step(name: str) -> int | None:
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-", 1)[1])
        except ValueError:
            return None
    return None


def artifact_paths_from_layout(
    layout: ArtifactOutputLayout,
) -> dict[str, Path | None]:
    """Map ExecutionResult field names to canonical layout paths."""
    return {
        "adapter_path": layout.adapter_dir,
        "checkpoint_path": layout.adapter_checkpoints_dir,
        "merged_model_path": layout.merged_dir,
        "export_path": layout.export_dir,
        "metrics_path": layout.experiment_metrics_dir,
        "logs_path": layout.experiment_logs_dir,
    }


__all__ = [
    "ArtifactOutputManager",
    "artifact_paths_from_layout",
    "requires_drive_workspace",
    "resolve_experiment_id",
    "resolve_workspace_root",
    "should_use_canonical_layout",
    "validate_drive_workspace_contract",
]
