"""Canonical Drive workspace layout for production artifact outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactOutputLayout:
    """
    Single source of truth for experiment artifact destinations.

    All paths are under ``workspace_root`` (the AIODOO Drive workspace).
    Repository source code must never be used as an output root.
    """

    workspace_root: Path
    experiment_id: str

    @property
    def adapter_dir(self) -> Path:
        """Final published adapter: ``models/adapters/{EXP}/``."""
        return self.workspace_root / "models" / "adapters" / self.experiment_id

    @property
    def adapter_checkpoints_dir(self) -> Path:
        """Training checkpoints: ``training/cache/{EXP}/checkpoints/``."""
        return self.workspace_root / "training" / "cache" / self.experiment_id / "checkpoints"

    @property
    def merged_dir(self) -> Path:
        """Merged model weights: ``models/merged/{EXP}/``."""
        return self.workspace_root / "models" / "merged" / self.experiment_id

    @property
    def export_dir(self) -> Path:
        """Export bundles: ``models/exports/{EXP}/``."""
        return self.workspace_root / "models" / "exports" / self.experiment_id

    @property
    def experiment_dir(self) -> Path:
        """Experiment metadata root: ``experiments/{EXP}/``."""
        return self.workspace_root / "experiments" / self.experiment_id

    @property
    def experiment_config_dir(self) -> Path:
        return self.experiment_dir / "config"

    @property
    def experiment_metrics_dir(self) -> Path:
        return self.experiment_dir / "metrics"

    @property
    def experiment_validation_dir(self) -> Path:
        return self.experiment_dir / "validation"

    @property
    def experiment_logs_dir(self) -> Path:
        return self.experiment_dir / "logs"

    @property
    def metrics_history_path(self) -> Path:
        return self.experiment_metrics_dir / "history.jsonl"

    @property
    def tracking_root(self) -> Path:
        return self.experiment_logs_dir / "tracking"

    @property
    def summary_path(self) -> Path:
        return self.experiment_dir / "summary.json"

    @property
    def readme_path(self) -> Path:
        return self.experiment_dir / "README.md"

    @property
    def adapter_manifest_path(self) -> Path:
        return self.adapter_dir / "manifest.json"

    @property
    def merged_manifest_path(self) -> Path:
        return self.merged_dir / "manifest.json"

    @property
    def export_manifest_path(self) -> Path:
        return self.export_dir / "manifest.json"

    def as_path_map(self) -> dict[str, Path]:
        """Return all canonical paths for diagnostics and config overlay."""
        return {
            "workspace_root": self.workspace_root,
            "adapter_dir": self.adapter_dir,
            "adapter_checkpoints_dir": self.adapter_checkpoints_dir,
            "merged_dir": self.merged_dir,
            "export_dir": self.export_dir,
            "experiment_dir": self.experiment_dir,
            "experiment_config_dir": self.experiment_config_dir,
            "experiment_metrics_dir": self.experiment_metrics_dir,
            "experiment_validation_dir": self.experiment_validation_dir,
            "experiment_logs_dir": self.experiment_logs_dir,
            "metrics_history_path": self.metrics_history_path,
            "tracking_root": self.tracking_root,
            "summary_path": self.summary_path,
            "readme_path": self.readme_path,
        }


__all__ = ["ArtifactOutputLayout"]
