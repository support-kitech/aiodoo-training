"""Phase 4 export manifest domain — bundle inventory and compatibility."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from aiodoo_training.domain.artifacts import ExportArtifact
from aiodoo_training.domain.enums import ExportType
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.training_policies import TRAINING_PROTOCOL_VERSION

ARTIFACT_PROTOCOL_VERSION = "1"
EXPORT_MANIFEST_SCHEMA_VERSION = "1"


class ArtifactValidationPolicy(StrEnum):
    """Producer-side integrity severity for artifact bundles."""

    STRICT = "strict"
    WARN = "warn"
    RELAXED = "relaxed"


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    """Logical metadata for one file inside an artifact bundle."""

    role: str
    relative_path: str
    checksum: str
    content_type: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.role or not self.role.strip():
            raise ValueError("ArtifactDescriptor.role must be non-empty.")
        if not self.relative_path or not self.relative_path.strip():
            raise ValueError("ArtifactDescriptor.relative_path must be non-empty.")
        if not self.checksum or not self.checksum.strip():
            raise ValueError("ArtifactDescriptor.checksum must be non-empty.")


@dataclass(frozen=True, slots=True)
class ArtifactIndexEntry:
    """One published bundle row discoverable without reading every file tree."""

    bundle_path: str
    experiment_id: ExperimentId
    run_id: RunId
    export_fingerprint: str
    artifact_protocol_version: str
    export_types: tuple[str, ...]
    roles: tuple[str, ...]
    created_at: datetime | None = None
    manifest_relpath: str = "export_manifest.json"

    def __post_init__(self) -> None:
        if not self.bundle_path or not self.bundle_path.strip():
            raise ValueError("ArtifactIndexEntry.bundle_path must be non-empty.")
        if not self.export_fingerprint or not self.export_fingerprint.strip():
            raise ValueError("ArtifactIndexEntry.export_fingerprint must be non-empty.")
        if not self.artifact_protocol_version or not self.artifact_protocol_version.strip():
            raise ValueError("ArtifactIndexEntry.artifact_protocol_version must be non-empty.")


@dataclass(frozen=True, slots=True)
class ExportManifest:
    """Per-bundle inventory: logical roles, relative paths, checksums, protocols."""

    schema_version: str
    artifact_protocol_version: str
    experiment_id: ExperimentId
    run_id: RunId
    model_fingerprint: str
    adapter_fingerprint: str
    config_fingerprint: str
    export_backend_key: str
    export_types: tuple[str, ...]
    artifacts: tuple[ArtifactDescriptor, ...]
    required_artifacts: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    export_fingerprint: str = ""
    evaluation_fingerprint: str | None = None
    training_protocol_version: str = TRAINING_PROTOCOL_VERSION
    software: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.schema_version or not self.schema_version.strip():
            raise ValueError("ExportManifest.schema_version must be non-empty.")
        if not self.artifact_protocol_version or not self.artifact_protocol_version.strip():
            raise ValueError("ExportManifest.artifact_protocol_version must be non-empty.")
        object.__setattr__(
            self,
            "software",
            MappingProxyType({str(k): str(v) for k, v in self.software.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_protocol_version": self.artifact_protocol_version,
            "experiment_id": self.experiment_id.value,
            "run_id": self.run_id.value,
            "model_fingerprint": self.model_fingerprint,
            "adapter_fingerprint": self.adapter_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "export_backend_key": self.export_backend_key,
            "export_types": list(self.export_types),
            "artifacts": [
                {
                    "role": d.role,
                    "relative_path": d.relative_path,
                    "checksum": d.checksum,
                    "content_type": d.content_type,
                    "required": d.required,
                }
                for d in self.artifacts
            ],
            "required_artifacts": list(self.required_artifacts),
            "artifact_paths": list(self.artifact_paths),
            "training_protocol_version": self.training_protocol_version,
            "software": dict(self.software),
            "export_fingerprint": self.export_fingerprint,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExportManifest:
        artifacts_raw = data.get("artifacts") or ()
        artifacts = tuple(
            ArtifactDescriptor(
                role=_role_value(d.get("role", "")),
                relative_path=str(d.get("relative_path", "")),
                checksum=str(d.get("checksum", "")),
                content_type=str(d["content_type"]) if d.get("content_type") is not None else None,
                required=bool(d.get("required", True)),
            )
            for d in artifacts_raw
        )
        created = data.get("created_at")
        return cls(
            schema_version=str(data.get("schema_version", EXPORT_MANIFEST_SCHEMA_VERSION)),
            artifact_protocol_version=str(
                data.get("artifact_protocol_version", ARTIFACT_PROTOCOL_VERSION)
            ),
            experiment_id=ExperimentId(value=str(data["experiment_id"])),
            run_id=RunId(value=str(data["run_id"])),
            model_fingerprint=str(data.get("model_fingerprint", "")),
            adapter_fingerprint=str(data.get("adapter_fingerprint", "")),
            config_fingerprint=str(data.get("config_fingerprint", "")),
            evaluation_fingerprint=(
                str(data["evaluation_fingerprint"])
                if data.get("evaluation_fingerprint") is not None
                else None
            ),
            export_backend_key=str(data.get("export_backend_key", "")),
            export_types=tuple(str(x) for x in (data.get("export_types") or ())),
            artifacts=artifacts,
            required_artifacts=tuple(str(x) for x in (data.get("required_artifacts") or ())),
            artifact_paths=tuple(str(x) for x in (data.get("artifact_paths") or ())),
            training_protocol_version=str(
                data.get("training_protocol_version", TRAINING_PROTOCOL_VERSION)
            ),
            software=dict(data.get("software") or {}),
            export_fingerprint=str(data.get("export_fingerprint", "")),
            created_at=datetime.fromisoformat(created) if isinstance(created, str) else None,
        )


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    """Top-level on-disk package handed to aiodoo-models."""

    root: Path
    manifest: ExportManifest
    artifacts: tuple[ExportArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactCompatibilityPolicy:
    """Consumer negotiation for artifact protocol versions and roles."""

    accepted_artifact_protocols: tuple[str, ...]
    required_roles: tuple[str, ...]
    optional_roles: tuple[str, ...] = ()
    reject_unknown_roles: bool = False

    def __post_init__(self) -> None:
        if not self.accepted_artifact_protocols:
            raise ValueError(
                "ArtifactCompatibilityPolicy.accepted_artifact_protocols must be non-empty."
            )


def compute_export_fingerprint(manifest_body: Mapping[str, Any]) -> str:
    """
    Digest of manifest contents excluding volatile identity fields.

    Excludes ``export_fingerprint`` and ``created_at`` so the digest is stable
    across publish retries on different machines.
    """
    payload = {
        k: v
        for k, v in manifest_body.items()
        if k not in {"export_fingerprint", "created_at"}
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _role_value(role: Any) -> str:
    if isinstance(role, ExportType):
        return role.value
    return str(role)
