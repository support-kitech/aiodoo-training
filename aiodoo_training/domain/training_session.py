"""Phase 3 training session domain — immutable COW run cursor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from aiodoo_training.domain.enums import TrainingStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.session import DatasetSession


def _freeze_str_map(value: Mapping[str, str] | None) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType({str(k): str(v) for k, v in value.items()})


@dataclass(frozen=True, slots=True)
class TrainingState:
    """Machine-readable lifecycle snapshot (maps to TrainingStatus)."""

    status: TrainingStatus
    global_step: int = 0
    epoch: float = 0.0
    message: str | None = None

    def __post_init__(self) -> None:
        if self.global_step < 0:
            raise ValueError("TrainingState.global_step must be >= 0.")
        if self.epoch < 0:
            raise ValueError("TrainingState.epoch must be >= 0.")


@dataclass(frozen=True, slots=True)
class TrainingSession:
    """
    Immutable identity + lifecycle cursor for a single training run.

    Updates use copy-on-write helpers. Never mutated in place.
    """

    session_id: str
    experiment_id: ExperimentId
    run_id: RunId
    status: TrainingStatus = TrainingStatus.PENDING
    global_step: int = 0
    epoch: float = 0.0
    max_steps: int | None = None
    dataset_session: DatasetSession | None = None
    execution_digest: str = ""
    model_fingerprint: str = ""
    adapter_fingerprint: str = ""
    checkpoint_fingerprint: str | None = None
    resume_from: Path | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("TrainingSession.session_id must be non-empty.")
        if self.global_step < 0:
            raise ValueError("TrainingSession.global_step must be >= 0.")
        if self.epoch < 0:
            raise ValueError("TrainingSession.epoch must be >= 0.")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("TrainingSession.max_steps must be >= 1 when set.")
        object.__setattr__(self, "metadata", _freeze_str_map(self.metadata))

    def with_status(self, status: TrainingStatus, *, message: str | None = None) -> TrainingSession:
        meta = dict(self.metadata)
        if message is not None:
            meta["status_message"] = message
        return replace(
            self,
            status=status,
            updated_at=datetime.now(UTC),
            metadata=MappingProxyType(meta),
        )

    def advance_step(self, *, steps: int = 1, epoch: float | None = None) -> TrainingSession:
        if steps < 0:
            raise ValueError("steps must be >= 0.")
        return replace(
            self,
            global_step=self.global_step + steps,
            epoch=self.epoch if epoch is None else epoch,
            updated_at=datetime.now(UTC),
        )

    def with_dataset_session(self, dataset_session: DatasetSession | None) -> TrainingSession:
        return replace(self, dataset_session=dataset_session, updated_at=datetime.now(UTC))

    def with_checkpoint(
        self, fingerprint: str | None, *, path: Path | None = None
    ) -> TrainingSession:
        return replace(
            self,
            checkpoint_fingerprint=fingerprint,
            resume_from=path if path is not None else self.resume_from,
            updated_at=datetime.now(UTC),
        )

    def to_state(self) -> TrainingState:
        return TrainingState(
            status=self.status,
            global_step=self.global_step,
            epoch=self.epoch,
            message=self.metadata.get("status_message"),
        )
