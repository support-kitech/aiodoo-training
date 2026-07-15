"""Checkpointing façades — re-exports and DatasetSession sidecar helpers."""

from aiodoo_training.training.checkpoint_manager import (
    DATASET_SESSION_FILENAME,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    RNG_FILENAME,
    CheckpointIndex,
    CheckpointManager,
    ResumeValidationContext,
    SaveCheckpointRequest,
    dataset_session_from_dict,
    dataset_session_to_dict,
)

__all__ = [
    "CheckpointIndex",
    "CheckpointManager",
    "DATASET_SESSION_FILENAME",
    "INDEX_FILENAME",
    "MANIFEST_FILENAME",
    "METRICS_FILENAME",
    "RNG_FILENAME",
    "ResumeValidationContext",
    "SaveCheckpointRequest",
    "dataset_session_from_dict",
    "dataset_session_to_dict",
]
