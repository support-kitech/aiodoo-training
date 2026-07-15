"""Phase 5 curriculum session domain — immutable COW curriculum cursor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType

from aiodoo_training.domain.enums import CurriculumStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId


def _freeze_str_map(value: Mapping[str, str] | None) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType({str(k): str(v) for k, v in value.items()})


@dataclass(frozen=True, slots=True)
class CurriculumState:
    """Machine-readable curriculum lifecycle snapshot."""

    status: CurriculumStatus
    stage_index: int = 0
    stage_count: int | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class CurriculumProgress:
    """Interim curriculum stage progress."""

    status: CurriculumStatus
    stage_index: int
    stage_count: int | None
    examples_in_stage: int = 0
    message: str | None = None


@dataclass(frozen=True, slots=True)
class CurriculumStatistics:
    """Immutable summary of a completed curriculum plan (not a runtime tracker)."""

    curriculum_fingerprint: str
    backend_key: str
    stage_count: int
    examples_total: int
    examples_per_stage: tuple[int, ...]
    stage_names: tuple[str, ...] = ()
    weight_per_stage: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class CurriculumSession:
    """Immutable identity + stage cursor for a curriculum plan."""

    session_id: str
    experiment_id: ExperimentId
    run_id: RunId
    status: CurriculumStatus = CurriculumStatus.PENDING
    stage_index: int = 0
    stage_count: int | None = None
    examples_in_stage: int = 0
    curriculum_fingerprint: str | None = None
    model_fingerprint: str = ""
    adapter_fingerprint: str = ""
    config_fingerprint: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("CurriculumSession.session_id must be non-empty.")
        object.__setattr__(self, "metadata", _freeze_str_map(self.metadata))

    def with_status(
        self, status: CurriculumStatus, *, message: str | None = None
    ) -> CurriculumSession:
        meta = dict(self.metadata)
        if message is not None:
            meta["status_message"] = message
        return replace(
            self,
            status=status,
            updated_at=datetime.now(UTC),
            metadata=MappingProxyType(meta),
        )

    def with_stage(
        self,
        *,
        stage_index: int | None = None,
        stage_count: int | None = None,
        examples_in_stage: int | None = None,
    ) -> CurriculumSession:
        return replace(
            self,
            stage_index=self.stage_index if stage_index is None else stage_index,
            stage_count=self.stage_count if stage_count is None else stage_count,
            examples_in_stage=(
                self.examples_in_stage if examples_in_stage is None else examples_in_stage
            ),
            updated_at=datetime.now(UTC),
        )

    def with_fingerprint(self, curriculum_fingerprint: str) -> CurriculumSession:
        return replace(
            self,
            curriculum_fingerprint=curriculum_fingerprint,
            updated_at=datetime.now(UTC),
        )

    def to_state(self) -> CurriculumState:
        return CurriculumState(
            status=self.status,
            stage_index=self.stage_index,
            stage_count=self.stage_count,
            message=self.metadata.get("status_message"),
        )

    def to_progress(self) -> CurriculumProgress:
        return CurriculumProgress(
            status=self.status,
            stage_index=self.stage_index,
            stage_count=self.stage_count,
            examples_in_stage=self.examples_in_stage,
            message=self.metadata.get("status_message"),
        )
