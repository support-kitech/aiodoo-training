"""Phase 4 export session domain — immutable COW export cursor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from aiodoo_training.domain.enums import ExportStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId


def _freeze_str_map(value: Mapping[str, str] | None) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType({str(k): str(v) for k, v in value.items()})


@dataclass(frozen=True, slots=True)
class ExportState:
    """Machine-readable lifecycle snapshot (maps to ExportStatus)."""

    status: ExportStatus
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ExportSession:
    """
    Immutable identity + lifecycle cursor for a single export attempt.

    Updates use copy-on-write helpers. Never mutated in place.
    """

    session_id: str
    experiment_id: ExperimentId
    run_id: RunId
    status: ExportStatus = ExportStatus.PENDING
    model_fingerprint: str = ""
    adapter_fingerprint: str = ""
    config_fingerprint: str = ""
    evaluation_fingerprint: str | None = None
    export_fingerprint: str | None = None
    bundle_path: Path | None = None
    export_backend_key: str = ""
    export_types: tuple[str, ...] = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("ExportSession.session_id must be non-empty.")
        object.__setattr__(self, "metadata", _freeze_str_map(self.metadata))

    def with_status(self, status: ExportStatus, *, message: str | None = None) -> ExportSession:
        meta = dict(self.metadata)
        if message is not None:
            meta["status_message"] = message
        return replace(
            self,
            status=status,
            updated_at=datetime.now(UTC),
            metadata=MappingProxyType(meta),
        )

    def with_fingerprints(
        self,
        *,
        export_fingerprint: str | None = None,
        evaluation_fingerprint: str | None = None,
    ) -> ExportSession:
        return replace(
            self,
            export_fingerprint=(
                export_fingerprint if export_fingerprint is not None else self.export_fingerprint
            ),
            evaluation_fingerprint=(
                evaluation_fingerprint
                if evaluation_fingerprint is not None
                else self.evaluation_fingerprint
            ),
            updated_at=datetime.now(UTC),
        )

    def with_bundle(
        self, bundle_path: Path | None, *, export_fingerprint: str | None = None
    ) -> ExportSession:
        return replace(
            self,
            bundle_path=bundle_path,
            export_fingerprint=(
                export_fingerprint if export_fingerprint is not None else self.export_fingerprint
            ),
            updated_at=datetime.now(UTC),
        )

    def to_state(self) -> ExportState:
        return ExportState(
            status=self.status,
            message=self.metadata.get("status_message"),
        )
