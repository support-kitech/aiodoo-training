"""Phase 4 evaluation session domain — immutable COW run cursor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType

from aiodoo_training.domain.enums import DatasetSplitKind, EvaluationStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.domain.training import MetricSnapshot


def _freeze_str_map(value: Mapping[str, str] | None) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType({str(k): str(v) for k, v in value.items()})


@dataclass(frozen=True, slots=True)
class EvaluationState:
    """Machine-readable lifecycle snapshot (maps to EvaluationStatus)."""

    status: EvaluationStatus
    examples_seen: int = 0
    examples_total: int | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.examples_seen < 0:
            raise ValueError("EvaluationState.examples_seen must be >= 0.")
        if self.examples_total is not None and self.examples_total < 0:
            raise ValueError("EvaluationState.examples_total must be >= 0 when set.")


@dataclass(frozen=True, slots=True)
class EvaluationProgress:
    """Immutable snapshot of evaluation progress and interim metrics."""

    status: EvaluationStatus
    examples_seen: int
    examples_total: int | None
    metrics: tuple[MetricSnapshot, ...] = ()
    message: str | None = None

    def __post_init__(self) -> None:
        if self.examples_seen < 0:
            raise ValueError("EvaluationProgress.examples_seen must be >= 0.")
        if self.examples_total is not None and self.examples_total < 0:
            raise ValueError("EvaluationProgress.examples_total must be >= 0 when set.")


@dataclass(frozen=True, slots=True)
class EvaluationSession:
    """
    Immutable identity + lifecycle cursor for a single evaluation run.

    Updates use copy-on-write helpers. Never mutated in place.
    """

    session_id: str
    experiment_id: ExperimentId
    run_id: RunId
    status: EvaluationStatus = EvaluationStatus.PENDING
    examples_seen: int = 0
    examples_total: int | None = None
    dataset_session: DatasetSession | None = None
    split: DatasetSplitKind = DatasetSplitKind.VALIDATION
    model_fingerprint: str = ""
    adapter_fingerprint: str = ""
    config_fingerprint: str = ""
    execution_digest: str = ""
    evaluation_fingerprint: str | None = None
    report_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("EvaluationSession.session_id must be non-empty.")
        if self.examples_seen < 0:
            raise ValueError("EvaluationSession.examples_seen must be >= 0.")
        if self.examples_total is not None and self.examples_total < 0:
            raise ValueError("EvaluationSession.examples_total must be >= 0 when set.")
        object.__setattr__(self, "metadata", _freeze_str_map(self.metadata))

    def with_status(
        self, status: EvaluationStatus, *, message: str | None = None
    ) -> EvaluationSession:
        meta = dict(self.metadata)
        if message is not None:
            meta["status_message"] = message
        return replace(
            self,
            status=status,
            updated_at=datetime.now(UTC),
            metadata=MappingProxyType(meta),
        )

    def advance(self, *, examples: int = 1) -> EvaluationSession:
        if examples < 0:
            raise ValueError("examples must be >= 0.")
        return replace(
            self,
            examples_seen=self.examples_seen + examples,
            updated_at=datetime.now(UTC),
        )

    def with_dataset_session(self, dataset_session: DatasetSession | None) -> EvaluationSession:
        return replace(self, dataset_session=dataset_session, updated_at=datetime.now(UTC))

    def with_report(
        self,
        *,
        report_id: str,
        evaluation_fingerprint: str | None = None,
    ) -> EvaluationSession:
        if not report_id or not report_id.strip():
            raise ValueError("report_id must be non-empty.")
        return replace(
            self,
            report_id=report_id,
            evaluation_fingerprint=(
                evaluation_fingerprint
                if evaluation_fingerprint is not None
                else self.evaluation_fingerprint
            ),
            updated_at=datetime.now(UTC),
        )

    def to_state(self) -> EvaluationState:
        return EvaluationState(
            status=self.status,
            examples_seen=self.examples_seen,
            examples_total=self.examples_total,
            message=self.metadata.get("status_message"),
        )

    def to_progress(self, *, metrics: tuple[MetricSnapshot, ...] = ()) -> EvaluationProgress:
        return EvaluationProgress(
            status=self.status,
            examples_seen=self.examples_seen,
            examples_total=self.examples_total,
            metrics=metrics,
            message=self.metadata.get("status_message"),
        )
