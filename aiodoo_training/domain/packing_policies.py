"""Phase 5 packing / sampling / memory policy domain DTOs."""

from __future__ import annotations

from dataclasses import dataclass

from aiodoo_training.domain.enums import PackingMode, PackingOverflow


@dataclass(frozen=True, slots=True)
class PackingPolicy:
    """Declarative packing policy resolved from PackingSpec + config fragments."""

    backend_key: str = "none"
    mode: PackingMode = PackingMode.NONE
    max_sequence_length: int = 2048
    max_examples_per_sequence: int | None = None
    separator_token_id: int | None = None
    overflow: PackingOverflow = PackingOverflow.DEFER
    drop_last: bool = False
    seed: int | None = 42
    pad_to_multiple_of: int | None = None

    def __post_init__(self) -> None:
        if self.max_sequence_length < 1:
            raise ValueError("PackingPolicy.max_sequence_length must be >= 1.")
        if self.max_examples_per_sequence is not None and self.max_examples_per_sequence < 1:
            raise ValueError("PackingPolicy.max_examples_per_sequence must be >= 1.")


@dataclass(frozen=True, slots=True)
class MemoryPackingPolicy:
    """Memory-oriented packing preferences (consult ExecutionEnvironment)."""

    target_tokens_per_batch: int | None = None
    max_padding_ratio: float | None = None
    pad_to_multiple_of: int | None = None
    prefer_length_buckets: bool = False
    enable_packed_attention_hints: bool = False


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    """Additive sampling configuration (Phase 5)."""

    backend_key: str = "identity"
    seed: int | None = 42
    temperature: float = 1.0
    strata_key: str = "dataset_type"
    weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("SamplingSpec.temperature must be > 0.")
