"""Phase 5 packing configuration fragments."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aiodoo_training.domain.enums import PackingMode, PackingOverflow
from aiodoo_training.domain.packing_policies import MemoryPackingPolicy, PackingPolicy
from aiodoo_training.exceptions import ConfigError


class PackingFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "none"
    mode: Literal["none", "concat", "best_fit", "length_aware"] = "none"
    max_sequence_length: int = Field(default=2048, ge=1)
    max_examples_per_sequence: int | None = Field(default=None, ge=1)
    separator_token_id: int | None = None
    overflow: Literal["defer", "truncate", "reject"] = "defer"
    drop_last: bool = False
    seed: int | None = 42
    pad_to_multiple_of: int | None = Field(default=None, ge=1)

    @field_validator("backend")
    @classmethod
    def _backend_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("packing.backend must be non-empty")
        return value


class MemoryPackingFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_tokens_per_batch: int | None = Field(default=None, ge=1)
    max_padding_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    pad_to_multiple_of: int | None = Field(default=None, ge=1)
    prefer_length_buckets: bool = False
    enable_packed_attention_hints: bool = False


def parse_packing_config(raw: dict[str, Any] | None) -> PackingFragment:
    try:
        return PackingFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001 — wrap pydantic
        raise ConfigError(f"Invalid packing config: {exc}") from exc


def parse_memory_packing_config(raw: dict[str, Any] | None) -> MemoryPackingFragment:
    try:
        return MemoryPackingFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001 — wrap pydantic
        raise ConfigError(f"Invalid memory packing config: {exc}") from exc


def to_packing_policy(fragment: PackingFragment) -> PackingPolicy:
    return PackingPolicy(
        backend_key=fragment.backend,
        mode=PackingMode(fragment.mode),
        max_sequence_length=fragment.max_sequence_length,
        max_examples_per_sequence=fragment.max_examples_per_sequence,
        separator_token_id=fragment.separator_token_id,
        overflow=PackingOverflow(fragment.overflow),
        drop_last=fragment.drop_last,
        seed=fragment.seed,
        pad_to_multiple_of=fragment.pad_to_multiple_of,
    )


def to_memory_packing_policy(fragment: MemoryPackingFragment) -> MemoryPackingPolicy:
    return MemoryPackingPolicy(
        target_tokens_per_batch=fragment.target_tokens_per_batch,
        max_padding_ratio=fragment.max_padding_ratio,
        pad_to_multiple_of=fragment.pad_to_multiple_of,
        prefer_length_buckets=fragment.prefer_length_buckets,
        enable_packed_attention_hints=fragment.enable_packed_attention_hints,
    )


def validate_phase5_packing_fragments(fragment: PackingFragment) -> None:
    if fragment.backend not in {"none", "concat", "best_fit", "length_aware"}:
        # Allow unknown backends (registration-first) but modes must be known.
        pass
    if fragment.mode not in {"none", "concat", "best_fit", "length_aware"}:
        raise ConfigError(f"Unsupported packing.mode: {fragment.mode!r}")
