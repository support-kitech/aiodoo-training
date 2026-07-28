"""Canonical Drive workspace layout for production artifact outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aiodoo_training.naming import adapter_product_id, normalize_training_id


@dataclass(frozen=True, slots=True)
class ArtifactOutputLayout:
    """
    Single source of truth for training artifact destinations.

    All paths are under ``workspace_root`` (the AIODOO Drive workspace).
    Repository source code must never be used as an output root.

    Path roles:
    - ``training_id`` (e.g. ``coding``): experiment metadata folders
    - ``adapter_id`` (e.g. ``aiodoo-coding``): published adapters / merged / exports
    - ``cache_id`` (defaults to ``training_id``): checkpoint cache folder
      (use a distinct cache id when the same training id is trained on two bases,
      e.g. ``context-qwen`` / ``context-deepseek``)
    """

    workspace_root: Path
    training_id: str
    adapter_id: str
    cache_id: str = ""

    @classmethod
    def for_training(
        cls,
        workspace_root: Path,
        training_id: str,
        *,
        adapter_id: str | None = None,
        cache_id: str | None = None,
    ) -> ArtifactOutputLayout:
        """Build layout from a public training id (normalizes legacy EXP ids)."""
        tid = normalize_training_id(training_id)
        resolved_cache = (cache_id or "").strip() or tid
        return cls(
            workspace_root=workspace_root,
            training_id=tid,
            adapter_id=adapter_id or adapter_product_id(tid),
            cache_id=resolved_cache,
        )

    @property
    def experiment_id(self) -> str:
        """Public training id (alias kept for callers that still say experiment)."""
        return self.training_id

    @property
    def adapter_dir(self) -> Path:
        """Final published adapter: ``models/adapters/{aiodoo-<training>}/``."""
        return self.workspace_root / "models" / "adapters" / self.adapter_id

    @property
    def adapter_checkpoints_dir(self) -> Path:
        """Training checkpoints: ``training/cache/{cache_id}/checkpoints/``."""
        return self.workspace_root / "training" / "cache" / self.cache_id / "checkpoints"

    @property
    def merged_dir(self) -> Path:
        """Merged model weights: ``models/merged/{aiodoo-<training>}/``."""
        return self.workspace_root / "models" / "merged" / self.adapter_id

    @property
    def export_dir(self) -> Path:
        """Export bundles: ``models/exports/{aiodoo-<training>}/``."""
        return self.workspace_root / "models" / "exports" / self.adapter_id

    @property
    def experiment_dir(self) -> Path:
        """Run metadata root: ``experiments/{training_id}/``."""
        return self.workspace_root / "experiments" / self.training_id

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
