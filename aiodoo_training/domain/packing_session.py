"""Phase 5 packing session domain — immutable COW packing plan cursor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType

from aiodoo_training.domain.enums import PackingStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId


def _freeze_str_map(value: Mapping[str, str] | None) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType({str(k): str(v) for k, v in value.items()})


@dataclass(frozen=True, slots=True)
class PackingState:
    """Machine-readable packing lifecycle snapshot."""

    status: PackingStatus
    examples_seen: int = 0
    sequences_emitted: int = 0
    message: str | None = None


@dataclass(frozen=True, slots=True)
class PackingProgress:
    """Transient planning counters while status is PLANNING."""

    status: PackingStatus
    examples_seen: int
    sequences_emitted: int
    tokens_packed: int = 0
    tokens_padded: int = 0
    message: str | None = None


@dataclass(frozen=True, slots=True)
class PackedSpan:
    """Per-example span inside a packed sequence."""

    example_id: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class PackingStatistics:
    """Immutable summary of a completed packing plan (not a runtime tracker)."""

    packing_fingerprint: str
    backend_key: str
    examples_input: int
    examples_packed: int
    sequences_emitted: int
    tokens_content: int
    tokens_padded: int
    pad_ratio: float
    mean_examples_per_sequence: float
    max_sequence_length: int
    overflow_deferred: int = 0
    overflow_truncated: int = 0


@dataclass(frozen=True, slots=True)
class PackingSession:
    """Immutable identity + lifecycle cursor for one packing plan build."""

    session_id: str
    experiment_id: ExperimentId
    run_id: RunId
    status: PackingStatus = PackingStatus.PENDING
    examples_seen: int = 0
    sequences_emitted: int = 0
    tokens_packed: int = 0
    tokens_padded: int = 0
    packing_fingerprint: str | None = None
    model_fingerprint: str = ""
    adapter_fingerprint: str = ""
    config_fingerprint: str = ""
    execution_digest: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("PackingSession.session_id must be non-empty.")
        object.__setattr__(self, "metadata", _freeze_str_map(self.metadata))

    def with_status(self, status: PackingStatus, *, message: str | None = None) -> PackingSession:
        meta = dict(self.metadata)
        if message is not None:
            meta["status_message"] = message
        return replace(
            self,
            status=status,
            updated_at=datetime.now(UTC),
            metadata=MappingProxyType(meta),
        )

    def advance(
        self,
        *,
        examples: int = 0,
        sequences: int = 0,
        tokens_packed: int = 0,
        tokens_padded: int = 0,
    ) -> PackingSession:
        return replace(
            self,
            examples_seen=self.examples_seen + examples,
            sequences_emitted=self.sequences_emitted + sequences,
            tokens_packed=self.tokens_packed + tokens_packed,
            tokens_padded=self.tokens_padded + tokens_padded,
            updated_at=datetime.now(UTC),
        )

    def with_fingerprint(self, packing_fingerprint: str) -> PackingSession:
        return replace(
            self,
            packing_fingerprint=packing_fingerprint,
            updated_at=datetime.now(UTC),
        )

    def to_state(self) -> PackingState:
        return PackingState(
            status=self.status,
            examples_seen=self.examples_seen,
            sequences_emitted=self.sequences_emitted,
            message=self.metadata.get("status_message"),
        )

    def to_progress(self) -> PackingProgress:
        return PackingProgress(
            status=self.status,
            examples_seen=self.examples_seen,
            sequences_emitted=self.sequences_emitted,
            tokens_packed=self.tokens_packed,
            tokens_padded=self.tokens_padded,
            message=self.metadata.get("status_message"),
        )
