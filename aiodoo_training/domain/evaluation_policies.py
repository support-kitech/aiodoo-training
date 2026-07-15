"""Phase 4 evaluation and quality-gate domain policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from aiodoo_training.domain.enums import ComparisonOp, DatasetSplitKind


class ThresholdCombine(StrEnum):
    """How multiple quality thresholds combine into a pass/fail decision."""

    ALL = "all"
    ANY = "any"


class ThresholdSeverity(StrEnum):
    """Whether a threshold violation is a hard failure or a warning."""

    ERROR = "error"
    WARN = "warn"


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    """Declarative evaluation configuration (framework-independent)."""

    backend_key: str
    profile_key: str = "default"
    metrics: tuple[str, ...] = ()
    max_examples: int | None = None
    seed: int | None = None
    split: DatasetSplitKind = DatasetSplitKind.VALIDATION

    def __post_init__(self) -> None:
        if not self.backend_key or not self.backend_key.strip():
            raise ValueError("EvaluationPolicy.backend_key must be non-empty.")
        if self.max_examples is not None and self.max_examples < 1:
            raise ValueError("EvaluationPolicy.max_examples must be >= 1 when set.")


@dataclass(frozen=True, slots=True)
class QualityThreshold:
    """Single metric threshold for acceptance gating."""

    metric_key: str
    op: ComparisonOp
    value: float
    severity: ThresholdSeverity = ThresholdSeverity.ERROR

    def __post_init__(self) -> None:
        if not self.metric_key or not self.metric_key.strip():
            raise ValueError("QualityThreshold.metric_key must be non-empty.")


@dataclass(frozen=True, slots=True)
class AcceptancePolicy:
    """Combines quality thresholds into a ship/no-ship decision."""

    combine: ThresholdCombine = ThresholdCombine.ALL
    thresholds: tuple[QualityThreshold, ...] = ()
    require_pass_for_export: bool = False


@dataclass(frozen=True, slots=True)
class EvaluationProfile:
    """Declarative evaluation profile (metric set, limits)."""

    key: str
    metrics: tuple[str, ...] = ("loss", "perplexity", "token_accuracy")
    max_examples: int | None = None
    backend_key: str = "stub"

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("EvaluationProfile.key must be non-empty.")


@dataclass(frozen=True, slots=True)
class ExportProfile:
    """Declarative export profile (roles / layout)."""

    key: str
    export_types: tuple[str, ...] = (
        "peft_adapter",
        "tokenizer",
        "manifest",
        "model_card",
        "bundle",
    )
    backend_key: str = "stub"
    require_evaluation: bool = False

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("ExportProfile.key must be non-empty.")
