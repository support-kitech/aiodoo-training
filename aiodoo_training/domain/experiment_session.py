"""Phase 6 experiment session domain — immutable catalog cursor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType

from aiodoo_training.domain.enums import ExperimentStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.tracking_policies import TRACKING_PROTOCOL_VERSION


def _freeze_str_map(value: Mapping[str, str] | None) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType({str(k): str(v) for k, v in value.items()})


@dataclass(frozen=True, slots=True)
class ExperimentSession:
    """Immutable identity + catalog cursor for one logical experiment."""

    session_id: str
    experiment_id: ExperimentId
    name: str
    status: ExperimentStatus = ExperimentStatus.PENDING
    created_at: datetime | None = None
    updated_at: datetime | None = None
    config_fingerprint: str = ""
    model_fingerprint: str = ""
    adapter_fingerprint: str = ""
    latest_run_id: RunId | None = None
    run_count: int = 0
    tracking_protocol_version: str = TRACKING_PROTOCOL_VERSION
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("ExperimentSession.session_id must be non-empty.")
        if not self.name or not self.name.strip():
            raise ValueError("ExperimentSession.name must be non-empty.")
        if not self.tracking_protocol_version or not self.tracking_protocol_version.strip():
            raise ValueError("ExperimentSession.tracking_protocol_version must be non-empty.")
        object.__setattr__(self, "metadata", _freeze_str_map(self.metadata))

    def with_status(
        self, status: ExperimentStatus, *, message: str | None = None
    ) -> ExperimentSession:
        meta = dict(self.metadata)
        if message is not None:
            meta["status_message"] = message
        return replace(
            self,
            status=status,
            updated_at=datetime.now(UTC),
            metadata=MappingProxyType(meta),
        )

    def with_run(self, run_id: RunId) -> ExperimentSession:
        return replace(
            self,
            latest_run_id=run_id,
            run_count=self.run_count + 1,
            updated_at=datetime.now(UTC),
            status=(
                ExperimentStatus.ACTIVE if self.status is ExperimentStatus.PENDING else self.status
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperimentHistory:
    """Ordered run ids belonging to an experiment (observational)."""

    experiment_id: ExperimentId
    run_ids: tuple[RunId, ...] = ()

    def with_run(self, run_id: RunId) -> ExperimentHistory:
        if run_id in self.run_ids:
            return self
        return replace(self, run_ids=(*self.run_ids, run_id))

    def as_sequence(self) -> Sequence[RunId]:
        return self.run_ids
