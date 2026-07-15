"""Pipeline orchestration foundation (no concrete training stages)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from aiodoo_training.domain.config import ExperimentConfig
from aiodoo_training.domain.enums import PipelineStage, StageStatus, TrainingStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId, StageName
from aiodoo_training.domain.pipeline import PipelineResult, StageResult
from aiodoo_training.exceptions import PipelineError


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """
    Immutable context flowing through pipeline stages.

    Stages never mutate context in place. They return an updated copy via
    :meth:`with_values` (copy-on-write). Business state is stored under
    ``values`` using stable string keys agreed by collaborating stages.
    """

    experiment_id: ExperimentId | None = None
    run_id: RunId | None = None
    config: ExperimentConfig | None = None
    values: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def get(self, key: str, default: Any = None) -> Any:
        """Return a typed-agnostic value from the context bag."""
        return self.values.get(key, default)

    def with_values(self, **updates: Any) -> PipelineContext:
        """Return a new context with ``updates`` merged into ``values``."""
        merged = dict(self.values)
        merged.update(updates)
        return replace(self, values=MappingProxyType(merged))

    def with_identity(
        self,
        *,
        experiment_id: ExperimentId | None = None,
        run_id: RunId | None = None,
        config: ExperimentConfig | None = None,
    ) -> PipelineContext:
        """Return a new context with identity fields replaced."""
        return replace(
            self,
            experiment_id=self.experiment_id if experiment_id is None else experiment_id,
            run_id=self.run_id if run_id is None else run_id,
            config=self.config if config is None else config,
        )


class PipelineStageHandler(ABC):
    """
    Abstract handler for a single pipeline stage.

    Implementations belong outside this module (application / later phases).
    This layer must remain free of training or dataset business logic.
    """

    @property
    @abstractmethod
    def name(self) -> StageName:
        """Stable stage instance name (unique within a pipeline)."""

    @property
    @abstractmethod
    def stage(self) -> PipelineStage:
        """Stage kind from the domain enumeration."""

    @abstractmethod
    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        """Execute the stage and return an updated context plus result."""


class Pipeline:
    """
    Ordered orchestrator for :class:`PipelineStageHandler` instances.

    Phase 0 provides the framework only — no training stages are registered.
    An empty pipeline completes successfully with status ``COMPLETED``.

    Stages should be supplied at construction time. :meth:`add_stage` exists
    for composition before :meth:`run` and must not be used mid-execution.
    """

    def __init__(self, stages: Sequence[PipelineStageHandler] | None = None) -> None:
        self._stages: list[PipelineStageHandler] = list(stages or ())
        self._started = False

    @property
    def stages(self) -> tuple[PipelineStageHandler, ...]:
        """Immutable view of registered stage handlers."""
        return tuple(self._stages)

    def add_stage(self, stage: PipelineStageHandler) -> None:
        """Append a stage handler. Forbidden after :meth:`run` has started."""
        if self._started:
            raise PipelineError("Cannot add stages after Pipeline.run() has started.")
        self._stages.append(stage)

    def run(self, context: PipelineContext | None = None) -> PipelineResult:
        """Execute all stages sequentially; stop on first failure."""
        self._started = True
        current = context or PipelineContext()
        results: list[StageResult] = []

        for handler in self._stages:
            try:
                current, result = handler.run(current)
            except Exception as exc:  # noqa: BLE001 — capture into StageResult
                failed = StageResult(
                    name=handler.name,
                    stage=handler.stage,
                    status=StageStatus.FAILED,
                    message="Stage raised an exception.",
                    error=f"{type(exc).__name__}: {exc}",
                )
                results.append(failed)
                return PipelineResult(
                    experiment_id=current.experiment_id,
                    run_id=current.run_id,
                    status=TrainingStatus.FAILED,
                    stage_results=tuple(results),
                    message=f"Pipeline failed at stage '{handler.name.value}'.",
                )

            results.append(result)
            if result.status == StageStatus.FAILED:
                return PipelineResult(
                    experiment_id=current.experiment_id,
                    run_id=current.run_id,
                    status=TrainingStatus.FAILED,
                    stage_results=tuple(results),
                    message=f"Pipeline failed at stage '{handler.name.value}'.",
                )

        return PipelineResult(
            experiment_id=current.experiment_id,
            run_id=current.run_id,
            status=TrainingStatus.COMPLETED,
            stage_results=tuple(results),
            message="Pipeline completed successfully.",
        )


class NoOpStage(PipelineStageHandler):
    """Utility stage for tests and scaffolding — succeeds without side effects."""

    def __init__(self, name: str, stage: PipelineStage) -> None:
        self._name = StageName(value=name)
        self._stage = stage

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return self._stage

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        result = StageResult(
            name=self._name,
            stage=self._stage,
            status=StageStatus.SUCCEEDED,
            message="No-op stage completed.",
        )
        return context, result


def require_config(context: PipelineContext) -> ExperimentConfig:
    """Return ``context.config`` or raise when missing (helper for future stages)."""
    if context.config is None:
        raise PipelineError("PipelineContext.config is required for this stage.")
    return context.config
