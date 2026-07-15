"""Phase 4 evaluation configuration fragments (pydantic + domain mapping)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aiodoo_training.domain.enums import ComparisonOp, DatasetSplitKind
from aiodoo_training.domain.evaluation_policies import (
    AcceptancePolicy,
    EvaluationPolicy,
    QualityThreshold,
    ThresholdCombine,
    ThresholdSeverity,
)
from aiodoo_training.exceptions import ConfigError


class QualityThresholdFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_key: str
    op: Literal["ge", "le", "eq", "lt", "gt"] = "ge"
    value: float
    severity: Literal["error", "warn"] = "error"

    @field_validator("metric_key")
    @classmethod
    def _metric_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("metric_key must be non-empty")
        return value


class AcceptanceFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    combine: Literal["all", "any"] = "all"
    thresholds: list[QualityThresholdFragment] = Field(default_factory=list)
    require_pass_for_export: bool = False


class EvaluationFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "stub"
    profile: str = "default"
    metrics: list[str] = Field(
        default_factory=lambda: ["loss", "perplexity", "token_accuracy"]
    )
    max_examples: int | None = None
    seed: int | None = None
    split: Literal["validation", "test", "benchmark", "custom"] = "validation"
    enabled: bool = False
    acceptance: AcceptanceFragment = Field(default_factory=AcceptanceFragment)

    @field_validator("backend")
    @classmethod
    def _backend_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evaluation.backend must be non-empty")
        return value


def parse_evaluation_config(raw: dict[str, Any] | None) -> EvaluationFragment:
    try:
        return EvaluationFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001 — wrap pydantic
        raise ConfigError(f"Invalid evaluation config: {exc}") from exc


def to_evaluation_policy(fragment: EvaluationFragment) -> EvaluationPolicy:
    return EvaluationPolicy(
        backend_key=fragment.backend,
        profile_key=fragment.profile,
        metrics=tuple(fragment.metrics),
        max_examples=fragment.max_examples,
        seed=fragment.seed,
        split=DatasetSplitKind(fragment.split),
    )


_OP_ALIASES: dict[str, ComparisonOp] = {
    "ge": ComparisonOp.GE,
    "le": ComparisonOp.LE,
    "eq": ComparisonOp.EQ,
    "lt": ComparisonOp.LT,
    "gt": ComparisonOp.GT,
    ">=": ComparisonOp.GE,
    "<=": ComparisonOp.LE,
    "==": ComparisonOp.EQ,
    "<": ComparisonOp.LT,
    ">": ComparisonOp.GT,
}


def _parse_op(raw: str) -> ComparisonOp:
    key = raw.strip().lower() if raw.isalpha() else raw.strip()
    if key not in _OP_ALIASES:
        raise ConfigError(f"Unknown comparison op: {raw!r}")
    return _OP_ALIASES[key]


def to_acceptance_policy(fragment: EvaluationFragment | AcceptanceFragment) -> AcceptancePolicy:
    acceptance = (
        fragment.acceptance if isinstance(fragment, EvaluationFragment) else fragment
    )
    thresholds = tuple(
        QualityThreshold(
            metric_key=t.metric_key,
            op=_parse_op(t.op),
            value=t.value,
            severity=ThresholdSeverity(t.severity),
        )
        for t in acceptance.thresholds
    )
    return AcceptancePolicy(
        combine=ThresholdCombine(acceptance.combine),
        thresholds=thresholds,
        require_pass_for_export=acceptance.require_pass_for_export,
    )


def validate_phase4_evaluation_fragments(raw: dict[str, Any] | None) -> EvaluationFragment:
    """Validate and return the evaluation fragment (raises ConfigError)."""
    return parse_evaluation_config(raw)
