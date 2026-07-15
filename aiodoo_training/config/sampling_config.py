"""Phase 5 sampling configuration fragments."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aiodoo_training.domain.packing_policies import SamplingSpec
from aiodoo_training.exceptions import ConfigError


class SamplingFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "identity"
    seed: int | None = 42
    temperature: float = Field(default=1.0, gt=0)
    strata_key: str = "dataset_type"
    weights: list[float] = Field(default_factory=list)

    @field_validator("backend")
    @classmethod
    def _backend_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sampling.backend must be non-empty")
        return value


def parse_sampling_config(raw: dict[str, Any] | None) -> SamplingFragment:
    try:
        return SamplingFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001 — wrap pydantic
        raise ConfigError(f"Invalid sampling config: {exc}") from exc


def to_sampling_spec(fragment: SamplingFragment) -> SamplingSpec:
    return SamplingSpec(
        backend_key=fragment.backend,
        seed=fragment.seed,
        temperature=fragment.temperature,
        strata_key=fragment.strata_key,
        weights=tuple(fragment.weights),
    )


def validate_phase5_sampling_fragments(fragment: SamplingFragment) -> None:
    if fragment.temperature <= 0:
        raise ConfigError("sampling.temperature must be > 0")
