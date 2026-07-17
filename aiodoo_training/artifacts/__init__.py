"""Central artifact output pipeline for production Drive workspace."""

from aiodoo_training.artifacts.cleanup import (
    CleanupReport,
    cleanup_workspace,
    scan_abandoned_checkpoints,
    scan_empty_directories,
    scan_stale_tmp_dirs,
)
from aiodoo_training.artifacts.io_utils import ensure_parent_dir
from aiodoo_training.artifacts.output_layout import ArtifactOutputLayout
from aiodoo_training.artifacts.output_manager import (
    ArtifactOutputManager,
    artifact_paths_from_layout,
    resolve_experiment_id,
    resolve_workspace_root,
    should_use_canonical_layout,
)

__all__ = [
    "ArtifactOutputLayout",
    "ArtifactOutputManager",
    "CleanupReport",
    "artifact_paths_from_layout",
    "cleanup_workspace",
    "ensure_parent_dir",
    "resolve_experiment_id",
    "resolve_workspace_root",
    "scan_abandoned_checkpoints",
    "scan_empty_directories",
    "scan_stale_tmp_dirs",
    "should_use_canonical_layout",
]
