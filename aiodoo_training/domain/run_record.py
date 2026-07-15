"""Phase 6 run record domain — immutable observational run mirror."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType

from aiodoo_training.domain.enums import RunState
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.tracking_policies import TRACKING_PROTOCOL_VERSION


def _freeze_str_map(value: Mapping[str, str] | None) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType({str(k): str(v) for k, v in value.items()})


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Portable labels / notes for a run (observational)."""

    tags: tuple[tuple[str, str], ...] = ()
    notes: str | None = None
    parent_run_id: RunId | None = None
    resume_of: RunId | None = None
    extra: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_str_map(self.extra))


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Immutable observational record of one pipeline execution."""

    run_id: RunId
    experiment_id: ExperimentId
    state: RunState = RunState.PENDING
    training_session_id: str | None = None
    evaluation_session_id: str | None = None
    export_session_id: str | None = None
    packing_fingerprint: str | None = None
    curriculum_fingerprint: str | None = None
    checkpoint_refs: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    provenance_digest: str = ""
    tracking_protocol_version: str = TRACKING_PROTOCOL_VERSION
    started_at: datetime | None = None
    ended_at: datetime | None = None
    metadata: RunMetadata = field(default_factory=RunMetadata)

    def __post_init__(self) -> None:
        if not self.tracking_protocol_version or not self.tracking_protocol_version.strip():
            raise ValueError("RunRecord.tracking_protocol_version must be non-empty.")

    def with_state(self, state: RunState) -> RunRecord:
        ended = self.ended_at
        if state in {
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.ABORTED,
        }:
            ended = datetime.now(UTC)
        return replace(self, state=state, ended_at=ended)

    def with_fingerprints(
        self,
        *,
        packing_fingerprint: str | None = None,
        curriculum_fingerprint: str | None = None,
        provenance_digest: str | None = None,
    ) -> RunRecord:
        return replace(
            self,
            packing_fingerprint=(
                self.packing_fingerprint
                if packing_fingerprint is None
                else packing_fingerprint
            ),
            curriculum_fingerprint=(
                self.curriculum_fingerprint
                if curriculum_fingerprint is None
                else curriculum_fingerprint
            ),
            provenance_digest=(
                self.provenance_digest if provenance_digest is None else provenance_digest
            ),
        )

    def with_refs(
        self,
        *,
        checkpoint_refs: tuple[str, ...] | None = None,
        artifact_refs: tuple[str, ...] | None = None,
    ) -> RunRecord:
        return replace(
            self,
            checkpoint_refs=self.checkpoint_refs if checkpoint_refs is None else checkpoint_refs,
            artifact_refs=self.artifact_refs if artifact_refs is None else artifact_refs,
        )
