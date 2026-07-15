"""Training progress and checkpoint domain objects."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aiodoo_training.domain.enums import CheckpointType, TrainingStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """Point-in-time metric observation during training or evaluation."""

    name: str
    value: float
    step: int
    timestamp: datetime | None = None
    tags: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    """Mutable-looking but immutable snapshot of trainer progress."""

    status: TrainingStatus
    global_step: int
    epoch: float
    metrics: tuple[MetricSnapshot, ...] = ()
    message: str | None = None


@dataclass(frozen=True, slots=True)
class CheckpointHandle:
    """Opaque handle to a persisted checkpoint directory."""

    path: Path
    experiment_id: ExperimentId
    run_id: RunId
    checkpoint_type: CheckpointType
    global_step: int
    created_at: datetime | None = None
    metadata: tuple[tuple[str, str], ...] = ()
