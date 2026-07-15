"""Phase 2 configuration mapping — model / adaptation / execution / quantization."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aiodoo_training.domain.config import AdaptationSpec, ExecutionSpec
from aiodoo_training.domain.enums import (
    AcceleratorKind,
    AdapterType,
    DeviceKind,
    ModelFamily,
    Precision,
)
from aiodoo_training.domain.quantization import QuantizationPolicy
from aiodoo_training.domain.refs import ModelRef
from aiodoo_training.domain.resources import DevicePolicy, MemoryPolicy, PrecisionPolicy
from aiodoo_training.exceptions import ConfigError


class ModelConfigModel(BaseModel):
    """Validated base-model fragment."""

    model_config = ConfigDict(extra="forbid")

    identifier: str = Field(min_length=1)
    family: ModelFamily = ModelFamily.UNKNOWN
    revision: str | None = None
    local_path: str | None = None
    precision: Precision = Precision.BF16
    tokenizer_binding: str | None = None
    backend: str = "stub"


class AdaptationConfigModel(BaseModel):
    """Validated adaptation fragment."""

    model_config = ConfigDict(extra="allow")

    adapter_type: AdapterType = AdapterType.LORA
    rank: int | None = 8
    alpha: int | None = 16
    dropout: float | None = 0.05
    target_modules: list[str] = Field(default_factory=list)
    strategy: str | None = None

    @field_validator("rank")
    @classmethod
    def _rank_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("rank must be >= 1")
        return value


class QuantizationConfigModel(BaseModel):
    """Validated quantization fragment (library-agnostic)."""

    model_config = ConfigDict(extra="forbid")

    compute: Precision = Precision.BF16
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    @model_validator(mode="after")
    def _exclusive_bits(self) -> QuantizationConfigModel:
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot set both load_in_4bit and load_in_8bit")
        return self


class ExecutionConfigModel(BaseModel):
    """Validated execution / resource fragment."""

    model_config = ConfigDict(extra="forbid")

    preferred_device: DeviceKind = DeviceKind.AUTO
    allow_cpu_fallback: bool = True
    device_ids: list[int] = Field(default_factory=list)
    compute: Precision = Precision.BF16
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    activation_checkpointing: bool = False
    allow_cpu_offload: bool = False
    max_memory_gb: float | None = None
    accelerator: AcceleratorKind = AcceleratorKind.NONE
    resource_planner: str = "static"

    @model_validator(mode="after")
    def _exclusive_bits(self) -> ExecutionConfigModel:
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot set both load_in_4bit and load_in_8bit")
        return self


def parse_model_config(data: dict[str, Any]) -> ModelConfigModel:
    """Validate a model mapping; raise ConfigError on failure."""
    try:
        return ModelConfigModel.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid model config: {exc}") from exc


def parse_adaptation_config(data: dict[str, Any]) -> AdaptationConfigModel:
    """Validate an adaptation mapping."""
    try:
        return AdaptationConfigModel.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid adaptation config: {exc}") from exc


def parse_execution_config(data: dict[str, Any]) -> ExecutionConfigModel:
    """Validate an execution mapping (supports nested or flat forms)."""
    flat = _flatten_execution(data)
    try:
        return ExecutionConfigModel.model_validate(flat)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid execution config: {exc}") from exc


def parse_quantization_config(data: dict[str, Any]) -> QuantizationConfigModel:
    """Validate a quantization mapping."""
    try:
        return QuantizationConfigModel.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid quantization config: {exc}") from exc


def _flatten_execution(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize nested ExecutionSpec-shaped YAML into flat ExecutionConfigModel fields."""
    if not data:
        return {}
    # Nested form from example.yaml
    if "device" in data or "precision" in data or "memory" in data:
        device = data.get("device") or {}
        precision = data.get("precision") or {}
        memory = data.get("memory") or {}
        return {
            "preferred_device": device.get("preferred", "auto"),
            "allow_cpu_fallback": device.get("allow_cpu_fallback", True),
            "device_ids": device.get("device_ids", []),
            "compute": precision.get("compute", "bf16"),
            "load_in_4bit": precision.get("load_in_4bit", False),
            "load_in_8bit": precision.get("load_in_8bit", False),
            "activation_checkpointing": memory.get("activation_checkpointing", False),
            "allow_cpu_offload": memory.get("allow_cpu_offload", False),
            "max_memory_gb": memory.get("max_memory_gb"),
            "accelerator": data.get("accelerator", "none"),
            "resource_planner": data.get("resource_planner", "static"),
        }
    return dict(data)


def to_model_ref(model: ModelConfigModel) -> ModelRef:
    """Map validated model config to ModelRef."""
    return ModelRef(
        identifier=model.identifier,
        family=model.family,
        revision=model.revision,
        local_path=Path(model.local_path) if model.local_path else None,
        precision=model.precision,
    )


def to_adaptation_spec(adaptation: AdaptationConfigModel) -> AdaptationSpec:
    """Map validated adaptation config to AdaptationSpec."""
    extra: dict[str, Any] = {}
    if adaptation.strategy is not None:
        extra["strategy"] = adaptation.strategy
    return AdaptationSpec(
        adapter_type=adaptation.adapter_type,
        rank=adaptation.rank,
        alpha=adaptation.alpha,
        dropout=adaptation.dropout,
        target_modules=tuple(adaptation.target_modules),
        extra=MappingProxyType(extra),
    )


def to_execution_spec(execution: ExecutionConfigModel) -> ExecutionSpec:
    """Map validated execution config to ExecutionSpec."""
    return ExecutionSpec(
        device=DevicePolicy(
            preferred=execution.preferred_device,
            allow_cpu_fallback=execution.allow_cpu_fallback,
            device_ids=tuple(execution.device_ids),
        ),
        precision=PrecisionPolicy(
            compute=execution.compute,
            load_in_4bit=execution.load_in_4bit,
            load_in_8bit=execution.load_in_8bit,
        ),
        memory=MemoryPolicy(
            max_memory_gb=execution.max_memory_gb,
            activation_checkpointing=execution.activation_checkpointing,
            allow_cpu_offload=execution.allow_cpu_offload,
        ),
        accelerator=execution.accelerator,
    )


def to_quantization_spec(quant: QuantizationConfigModel) -> QuantizationPolicy:
    """Map validated quantization config to QuantizationPolicy."""
    return QuantizationPolicy(
        compute=quant.compute,
        load_in_4bit=quant.load_in_4bit,
        load_in_8bit=quant.load_in_8bit,
    )


def strategy_key_for(adaptation: AdaptationConfigModel) -> str:
    """Resolve registry key for AdaptationStrategyFactory."""
    if adaptation.strategy:
        return adaptation.strategy
    return adaptation.adapter_type.value
