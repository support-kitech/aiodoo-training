"""Pipeline result domain types shared across orchestration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from aiodoo_training.domain.enums import PipelineStage, StageStatus, TrainingStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId, StageName


@dataclass(frozen=True, slots=True)
class StageResult:
    """Immutable outcome of a single pipeline stage."""

    name: StageName
    stage: PipelineStage
    status: StageStatus
    message: str | None = None
    payload: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Aggregate outcome of a full pipeline execution."""

    experiment_id: ExperimentId | None
    run_id: RunId | None
    status: TrainingStatus
    stage_results: tuple[StageResult, ...] = ()
    message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == TrainingStatus.COMPLETED
