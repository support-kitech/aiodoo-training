"""Phase 5 curriculum configuration fragments."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aiodoo_training.domain.config import CurriculumSpec
from aiodoo_training.domain.enums import CurriculumMode
from aiodoo_training.exceptions import ConfigError


class CurriculumFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "none"
    mode: Literal["none", "sequential", "weighted_mix", "difficulty", "random", "mixed"] = "none"
    stages: list[str] = Field(default_factory=list)
    seed: int | None = 42

    @field_validator("backend")
    @classmethod
    def _backend_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("curriculum.backend must be non-empty")
        return value


def parse_curriculum_config(raw: dict[str, Any] | None) -> CurriculumFragment:
    try:
        return CurriculumFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001 — wrap pydantic
        raise ConfigError(f"Invalid curriculum config: {exc}") from exc


def to_curriculum_spec(fragment: CurriculumFragment) -> CurriculumSpec:
    return CurriculumSpec(
        mode=CurriculumMode(fragment.mode),
        stages=tuple(fragment.stages),
    )


def validate_phase5_curriculum_fragments(fragment: CurriculumFragment) -> None:
    allowed = {"none", "sequential", "weighted_mix", "difficulty", "random", "mixed"}
    if fragment.mode not in allowed:
        raise ConfigError(f"Unsupported curriculum.mode: {fragment.mode!r}")
