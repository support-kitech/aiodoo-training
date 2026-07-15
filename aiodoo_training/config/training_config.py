"""Phase 3 additive training configuration fragments (pydantic + domain mapping)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aiodoo_training.domain.enums import Precision
from aiodoo_training.domain.training_policies import (
    CheckpointPolicy,
    GradientAccumulationPolicy,
    GradientClippingPolicy,
    LossScalingPolicy,
    MixedPrecisionPolicy,
    OptimizerPolicy,
    ResumePolicy,
    SchedulerPolicy,
)
from aiodoo_training.exceptions import ConfigError


class TrainingFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "stub"
    max_steps: int | None = None
    logging_steps: int = 10


class OptimizerFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    betas: list[float] | None = None

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("optimizer.name must be non-empty")
        return value


class SchedulerFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = 0.03
    total_steps: int | None = None


class ResumeFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Literal["strict", "warn", "relaxed"] = "strict"


class GradientFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_norm: float | None = 1.0
    accumulation_steps: int | None = None


class MetricsFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history_path: Path | None = None


class CheckpointingFragment(BaseModel):
    """Additive checkpoint fields layered on frozen CheckpointingSpec fields."""

    model_config = ConfigDict(extra="allow")

    save_steps: int = 500
    save_total_limit: int = 3
    save_on_failure: bool = False
    validate_on_load: bool = True
    retention: str = "keep_last"
    output_dir: Path | None = None
    resume_from: Path | None = None


class CallbacksFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[str] = Field(default_factory=lambda: ["logging"])


def parse_training_config(raw: dict[str, Any] | None) -> TrainingFragment:
    try:
        return TrainingFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid training config: {exc}") from exc


def parse_optimizer_config(raw: dict[str, Any] | None) -> OptimizerFragment:
    try:
        return OptimizerFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid optimizer config: {exc}") from exc


def parse_scheduler_config(raw: dict[str, Any] | None) -> SchedulerFragment:
    try:
        return SchedulerFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid scheduler config: {exc}") from exc


def parse_resume_config(raw: dict[str, Any] | None) -> ResumeFragment:
    try:
        return ResumeFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid resume config: {exc}") from exc


def parse_gradient_config(raw: dict[str, Any] | None) -> GradientFragment:
    try:
        return GradientFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid gradient config: {exc}") from exc


def parse_metrics_config(raw: dict[str, Any] | None) -> MetricsFragment:
    try:
        return MetricsFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid metrics config: {exc}") from exc


def parse_checkpointing_fragment(raw: dict[str, Any] | None) -> CheckpointingFragment:
    try:
        return CheckpointingFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid checkpointing config: {exc}") from exc


def parse_callbacks_config(raw: list[Any] | dict[str, Any] | None) -> list[str]:
    if raw is None:
        return ["logging"]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, dict):
        names = raw.get("names") or raw.get("callbacks") or []
        return [str(item) for item in names]
    raise ConfigError("callbacks must be a list of names or a mapping with 'names'.")


def to_optimizer_policy(fragment: OptimizerFragment) -> OptimizerPolicy:
    beta1 = fragment.beta1
    beta2 = fragment.beta2
    if fragment.betas is not None and len(fragment.betas) == 2:
        beta1 = float(fragment.betas[0])
        beta2 = float(fragment.betas[1])
    return OptimizerPolicy(
        name=fragment.name,
        learning_rate=fragment.learning_rate,
        weight_decay=fragment.weight_decay,
        beta1=beta1,
        beta2=beta2,
        eps=fragment.eps,
    )


def to_scheduler_policy(fragment: SchedulerFragment) -> SchedulerPolicy:
    return SchedulerPolicy(
        name=fragment.name,
        warmup_ratio=fragment.warmup_ratio,
        total_steps=fragment.total_steps,
    )


def to_resume_policy(fragment: ResumeFragment) -> ResumePolicy:
    return ResumePolicy(fragment.policy)


def to_checkpoint_policy(fragment: CheckpointingFragment) -> CheckpointPolicy:
    return CheckpointPolicy(
        save_steps=fragment.save_steps,
        save_total_limit=fragment.save_total_limit,
        save_on_failure=fragment.save_on_failure,
        validate_on_load=fragment.validate_on_load,
    )


def to_gradient_policies(
    fragment: GradientFragment,
    *,
    accumulation_fallback: int = 1,
    precision: Precision = Precision.BF16,
) -> tuple[
    GradientAccumulationPolicy,
    GradientClippingPolicy,
    MixedPrecisionPolicy,
    LossScalingPolicy,
]:
    steps = (
        fragment.accumulation_steps
        if fragment.accumulation_steps is not None
        else accumulation_fallback
    )
    return (
        GradientAccumulationPolicy(steps=max(1, int(steps))),
        GradientClippingPolicy(max_norm=fragment.max_norm),
        MixedPrecisionPolicy(precision=precision),
        LossScalingPolicy(enabled=precision is Precision.FP16),
    )


def validate_phase3_fragments(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate Phase 3 additive fragments on a composed raw config mapping.

    Returns a normalized bag of policies / keys for builders.
    """
    training = parse_training_config(
        data.get("training") if isinstance(data.get("training"), dict) else {}
    )
    optimizer = parse_optimizer_config(
        data.get("optimizer") if isinstance(data.get("optimizer"), dict) else {}
    )
    scheduler = parse_scheduler_config(
        data.get("scheduler") if isinstance(data.get("scheduler"), dict) else {}
    )
    resume = parse_resume_config(data.get("resume") if isinstance(data.get("resume"), dict) else {})
    gradient = parse_gradient_config(
        data.get("gradient") if isinstance(data.get("gradient"), dict) else {}
    )
    metrics = parse_metrics_config(
        data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    )
    checkpointing = parse_checkpointing_fragment(
        data.get("checkpointing") if isinstance(data.get("checkpointing"), dict) else {}
    )
    callbacks = parse_callbacks_config(data.get("callbacks"))

    if training.max_steps is not None and training.max_steps < 1:
        raise ConfigError("training.max_steps must be >= 1 when set.")

    resume_from = checkpointing.resume_from
    if resume_from is not None and not Path(resume_from).exists():
        # Soft check at validate time — path may resolve later relative to config dir.
        pass

    return {
        "training": training,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "resume": resume,
        "gradient": gradient,
        "metrics": metrics,
        "checkpointing": checkpointing,
        "callbacks": callbacks,
        "optimizer_policy": to_optimizer_policy(optimizer),
        "scheduler_policy": to_scheduler_policy(scheduler),
        "resume_policy": to_resume_policy(resume),
        "checkpoint_policy": to_checkpoint_policy(checkpointing),
        "trainer_backend_key": training.backend,
    }
