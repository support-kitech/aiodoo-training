"""Phase 4 export configuration fragments (pydantic + domain mapping)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aiodoo_training.domain.export_manifest import (
    ARTIFACT_PROTOCOL_VERSION,
    ArtifactCompatibilityPolicy,
    ArtifactValidationPolicy,
)
from aiodoo_training.exceptions import ConfigError


class ExportFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "stub"
    profile: str = "peft_default"
    enabled: bool = False
    output_dir: Path | None = None
    export_types: list[str] = Field(
        default_factory=lambda: [
            "peft_adapter",
            "tokenizer",
            "manifest",
            "model_card",
            "bundle",
        ]
    )
    require_evaluation: bool = False
    require_pass_for_export: bool = False
    validation_policy: Literal["strict", "warn", "relaxed"] = "strict"
    accepted_artifact_protocols: list[str] = Field(
        default_factory=lambda: [ARTIFACT_PROTOCOL_VERSION]
    )
    required_roles: list[str] = Field(
        default_factory=lambda: ["peft_adapter", "manifest"]
    )
    optional_roles: list[str] = Field(
        default_factory=lambda: ["tokenizer", "model_card", "evaluation_report"]
    )

    @field_validator("backend")
    @classmethod
    def _backend_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("export.backend must be non-empty")
        return value


def parse_export_config(raw: dict[str, Any] | None) -> ExportFragment:
    try:
        return ExportFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001 — wrap pydantic
        raise ConfigError(f"Invalid export config: {exc}") from exc


def to_validation_policy(fragment: ExportFragment) -> ArtifactValidationPolicy:
    return ArtifactValidationPolicy(fragment.validation_policy)


def to_compatibility_policy(fragment: ExportFragment) -> ArtifactCompatibilityPolicy:
    return ArtifactCompatibilityPolicy(
        accepted_artifact_protocols=tuple(fragment.accepted_artifact_protocols),
        required_roles=tuple(fragment.required_roles),
        optional_roles=tuple(fragment.optional_roles),
    )


def validate_phase4_export_fragments(raw: dict[str, Any] | None) -> ExportFragment:
    """Validate and return the export fragment (raises ConfigError)."""
    return parse_export_config(raw)
