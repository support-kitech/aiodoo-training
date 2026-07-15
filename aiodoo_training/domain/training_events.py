"""Phase 3 training event domain DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.training import CheckpointHandle, MetricSnapshot, TrainingProgress


class TrainingEventKind(StrEnum):
    TRAINING_STARTED = "training_started"
    EPOCH_STARTED = "epoch_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    LOSS_COMPUTED = "loss_computed"
    METRICS_AGGREGATED = "metrics_aggregated"
    CHECKPOINT_CREATED = "checkpoint_created"
    CHECKPOINT_PRUNED = "checkpoint_pruned"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    kind: TrainingEventKind
    experiment_id: ExperimentId
    run_id: RunId
    session_id: str
    timestamp: datetime
    global_step: int = 0
    epoch: float = 0.0
    loss: float | None = None
    metrics: tuple[MetricSnapshot, ...] = ()
    checkpoint: CheckpointHandle | None = None
    pruned: tuple[CheckpointHandle, ...] = ()
    progress: TrainingProgress | None = None
    error: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class TrainerResult:
    """Outcome wrapper around frozen TrainingProgress."""

    progress: TrainingProgress
    checkpoint: CheckpointHandle | None = None
    history_path: Path | None = None
