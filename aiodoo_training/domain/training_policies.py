"""Phase 3 resume / checkpoint / gradient / optimizer / scheduler policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from aiodoo_training.domain.enums import Precision


class ResumePolicy(StrEnum):
    """Compatibility severity for checkpoint resume validation."""

    STRICT = "strict"
    WARN = "warn"
    RELAXED = "relaxed"


TRAINING_PROTOCOL_VERSION = "1"


@dataclass(frozen=True, slots=True)
class OptimizerPolicy:
    """Declared optimizer configuration (framework-independent)."""

    name: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("OptimizerPolicy.name must be non-empty.")
        if self.learning_rate <= 0:
            raise ValueError("OptimizerPolicy.learning_rate must be > 0.")


@dataclass(frozen=True, slots=True)
class SchedulerPolicy:
    """Declared LR scheduler configuration."""

    name: str = "cosine"  # cosine | linear | constant
    warmup_ratio: float = 0.03
    total_steps: int | None = None

    def __post_init__(self) -> None:
        if self.name not in {"cosine", "linear", "constant"}:
            raise ValueError(f"Unsupported scheduler: {self.name}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("SchedulerPolicy.warmup_ratio must be in [0, 1).")


@dataclass(frozen=True, slots=True)
class GradientAccumulationPolicy:
    steps: int = 1

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("GradientAccumulationPolicy.steps must be >= 1.")


@dataclass(frozen=True, slots=True)
class GradientClippingPolicy:
    max_norm: float | None = 1.0


@dataclass(frozen=True, slots=True)
class MixedPrecisionPolicy:
    precision: Precision = Precision.BF16


@dataclass(frozen=True, slots=True)
class LossScalingPolicy:
    enabled: bool = False
    init_scale: float = 2.0**16


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    save_steps: int = 500
    save_total_limit: int = 3
    save_on_failure: bool = False
    validate_on_load: bool = True

    def __post_init__(self) -> None:
        if self.save_steps < 1:
            raise ValueError("CheckpointPolicy.save_steps must be >= 1.")
        if self.save_total_limit < 1:
            raise ValueError("CheckpointPolicy.save_total_limit must be >= 1.")
