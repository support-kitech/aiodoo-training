"""PEFT LoRA / QLoRA / Full fine-tune adaptation strategies (infrastructure only)."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from aiodoo_training.domain.adapter_info import AdapterCapabilities, AdapterMetadata
from aiodoo_training.domain.config import AdaptationSpec
from aiodoo_training.domain.enums import AdapterType
from aiodoo_training.domain.handles import BaseModelHandle, TrainableModelHandle
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.exceptions import DomainError
from aiodoo_training.infrastructure.model_handles import (
    OpaqueBaseModel,
    OpaqueTrainableModel,
    as_trainable_handle,
    require_base_carrier,
    require_trainable_carrier,
)
from aiodoo_training.ports.model import AdaptationStrategy


def _stub_apply(
    base: OpaqueBaseModel,
    spec: AdaptationSpec,
    *,
    strategy_key: str,
    requires_quantization: bool,
) -> TrainableModelHandle:
    """Apply a synthetic adapter to a stub framework model payload."""
    payload = base.framework_model
    if not isinstance(payload, dict) or payload.get("kind") != "stub":
        raise DomainError(f"Strategy '{strategy_key}' stub path requires StubModelBackend handles.")
    total = int(payload.get("num_parameters", 0))
    rank = spec.rank or 8
    # Deterministic synthetic trainable count (rank-scaled).
    trainable = min(total, max(1, rank * max(1, len(spec.target_modules)) * 12))
    adapted_payload = {
        **payload,
        "adapter_type": spec.adapter_type.value,
        "rank": rank,
        "trainable_parameters": trainable,
    }
    metadata = AdapterMetadata(
        adapter_type=spec.adapter_type,
        rank=spec.rank,
        alpha=spec.alpha,
        dropout=spec.dropout,
        target_modules=tuple(spec.target_modules),
        trainable_parameters=trainable,
        total_parameters=total,
        capabilities=AdapterCapabilities(
            supports_merge=spec.adapter_type != AdapterType.FULL,
            requires_quantization=requires_quantization,
            extra=MappingProxyType({"stub": "true"}),
        ),
        strategy_key=strategy_key,
    )
    carrier = OpaqueTrainableModel(
        framework_model=adapted_payload,
        aiodoo_adapter_metadata=metadata,
        base=base,
        strategy_key=strategy_key,
    )
    return as_trainable_handle(carrier)


def _peft_lora_config(spec: AdaptationSpec) -> Any:
    from peft import LoraConfig, TaskType

    rank = spec.rank if spec.rank is not None else 8
    alpha = spec.alpha if spec.alpha is not None else rank * 2
    dropout = spec.dropout if spec.dropout is not None else 0.05
    targets = list(spec.target_modules) or ["q_proj", "v_proj"]
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def _count_trainable(model: Any) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
    return trainable, total


class LoraAdaptationStrategy(AdaptationStrategy):
    """LoRA via PEFT when available; falls back to stub semantics for stub models."""

    STRATEGY_KEY = "lora"

    def apply(
        self,
        base_model: BaseModelHandle,
        spec: AdaptationSpec,
        execution: ExecutionEnvironment,
    ) -> TrainableModelHandle:
        _ = execution
        if spec.adapter_type != AdapterType.LORA:
            raise DomainError(
                f"LoraAdaptationStrategy expects adapter_type=lora; got {spec.adapter_type}."
            )
        base = require_base_carrier(base_model)
        if isinstance(base.framework_model, dict) and base.framework_model.get("kind") == "stub":
            return _stub_apply(
                base,
                AdaptationSpec(
                    adapter_type=AdapterType.LORA,
                    rank=spec.rank,
                    alpha=spec.alpha,
                    dropout=spec.dropout,
                    target_modules=spec.target_modules,
                    extra=spec.extra,
                ),
                strategy_key=self.STRATEGY_KEY,
                requires_quantization=False,
            )
        try:
            from peft import get_peft_model
        except ImportError as exc:  # pragma: no cover
            raise DomainError(
                "peft is required for LoRA on HuggingFace models. "
                "Install requirements/train.txt or use StubModelBackend."
            ) from exc

        peft_model = get_peft_model(base.framework_model, _peft_lora_config(spec))
        trainable, total = _count_trainable(peft_model)
        metadata = AdapterMetadata(
            adapter_type=AdapterType.LORA,
            rank=spec.rank,
            alpha=spec.alpha,
            dropout=spec.dropout,
            target_modules=tuple(spec.target_modules),
            trainable_parameters=trainable,
            total_parameters=total,
            capabilities=AdapterCapabilities(supports_merge=True),
            strategy_key=self.STRATEGY_KEY,
        )
        return as_trainable_handle(
            OpaqueTrainableModel(
                framework_model=peft_model,
                aiodoo_adapter_metadata=metadata,
                base=base,
                strategy_key=self.STRATEGY_KEY,
            )
        )

    def trainable_parameter_count(self, model: TrainableModelHandle) -> int:
        carrier = require_trainable_carrier(model)
        meta = carrier.aiodoo_adapter_metadata
        if meta.trainable_parameters is not None:
            return meta.trainable_parameters
        return _count_trainable(carrier.framework_model)[0]


class QLoraAdaptationStrategy(AdaptationStrategy):
    """
    QLoRA strategy — expects quantized base load via QuantizationPolicy (4-bit).

    PEFT LoRA is applied on top; quantization itself remains a load-time concern.
    """

    STRATEGY_KEY = "qlora"

    def apply(
        self,
        base_model: BaseModelHandle,
        spec: AdaptationSpec,
        execution: ExecutionEnvironment,
    ) -> TrainableModelHandle:
        if spec.adapter_type != AdapterType.QLORA:
            raise DomainError(
                f"QLoraAdaptationStrategy expects adapter_type=qlora; got {spec.adapter_type}."
            )
        if not (execution.precision_policy.load_in_4bit or execution.precision_policy.load_in_8bit):
            # Soft requirement: still allow stub path without quantization flags.
            base = require_base_carrier(base_model)
            fw = base.framework_model
            if isinstance(fw, dict) and fw.get("kind") == "stub":
                return _stub_apply(
                    base,
                    AdaptationSpec(
                        adapter_type=AdapterType.QLORA,
                        rank=spec.rank,
                        alpha=spec.alpha,
                        dropout=spec.dropout,
                        target_modules=spec.target_modules,
                        extra=spec.extra,
                    ),
                    strategy_key=self.STRATEGY_KEY,
                    requires_quantization=True,
                )
            raise DomainError(
                "QLoRA requires PrecisionPolicy.load_in_4bit or load_in_8bit "
                "on the resolved ExecutionEnvironment."
            )

        base = require_base_carrier(base_model)
        if isinstance(base.framework_model, dict) and base.framework_model.get("kind") == "stub":
            return _stub_apply(
                base,
                AdaptationSpec(
                    adapter_type=AdapterType.QLORA,
                    rank=spec.rank,
                    alpha=spec.alpha,
                    dropout=spec.dropout,
                    target_modules=spec.target_modules,
                    extra=spec.extra,
                ),
                strategy_key=self.STRATEGY_KEY,
                requires_quantization=True,
            )

        try:
            from peft import get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:  # pragma: no cover
            raise DomainError(
                "peft is required for QLoRA on HuggingFace models. "
                "Install requirements/train.txt or use StubModelBackend."
            ) from exc

        prepared = prepare_model_for_kbit_training(base.framework_model)
        peft_model = get_peft_model(prepared, _peft_lora_config(spec))
        trainable, total = _count_trainable(peft_model)
        metadata = AdapterMetadata(
            adapter_type=AdapterType.QLORA,
            rank=spec.rank,
            alpha=spec.alpha,
            dropout=spec.dropout,
            target_modules=tuple(spec.target_modules),
            trainable_parameters=trainable,
            total_parameters=total,
            capabilities=AdapterCapabilities(
                supports_merge=True,
                requires_quantization=True,
            ),
            strategy_key=self.STRATEGY_KEY,
        )
        return as_trainable_handle(
            OpaqueTrainableModel(
                framework_model=peft_model,
                aiodoo_adapter_metadata=metadata,
                base=base,
                strategy_key=self.STRATEGY_KEY,
            )
        )

    def trainable_parameter_count(self, model: TrainableModelHandle) -> int:
        carrier = require_trainable_carrier(model)
        meta = carrier.aiodoo_adapter_metadata
        if meta.trainable_parameters is not None:
            return meta.trainable_parameters
        return _count_trainable(carrier.framework_model)[0]


class FullFineTuneAdaptationStrategy(AdaptationStrategy):
    """
    Full fine-tuning placeholder.

    Marks all parameters trainable for stub models. Real Torch path freezes nothing.
    """

    STRATEGY_KEY = "full"

    def apply(
        self,
        base_model: BaseModelHandle,
        spec: AdaptationSpec,
        execution: ExecutionEnvironment,
    ) -> TrainableModelHandle:
        _ = execution
        if spec.adapter_type != AdapterType.FULL:
            raise DomainError(
                "FullFineTuneAdaptationStrategy expects adapter_type=full; "
                f"got {spec.adapter_type}."
            )
        base = require_base_carrier(base_model)
        if isinstance(base.framework_model, dict) and base.framework_model.get("kind") == "stub":
            return _stub_apply(
                base,
                AdaptationSpec(
                    adapter_type=AdapterType.FULL,
                    rank=None,
                    alpha=None,
                    dropout=None,
                    target_modules=(),
                    extra=spec.extra,
                ),
                strategy_key=self.STRATEGY_KEY,
                requires_quantization=False,
            )

        model = base.framework_model
        for param in model.parameters():
            param.requires_grad = True
        trainable, total = _count_trainable(model)
        metadata = AdapterMetadata(
            adapter_type=AdapterType.FULL,
            trainable_parameters=trainable,
            total_parameters=total,
            capabilities=AdapterCapabilities(supports_merge=False),
            strategy_key=self.STRATEGY_KEY,
            extra=MappingProxyType({"status": "placeholder"}),
        )
        return as_trainable_handle(
            OpaqueTrainableModel(
                framework_model=model,
                aiodoo_adapter_metadata=metadata,
                base=base,
                strategy_key=self.STRATEGY_KEY,
            )
        )

    def trainable_parameter_count(self, model: TrainableModelHandle) -> int:
        carrier = require_trainable_carrier(model)
        meta = carrier.aiodoo_adapter_metadata
        if meta.trainable_parameters is not None:
            return meta.trainable_parameters
        return _count_trainable(carrier.framework_model)[0]


def register_default_adaptation_strategies(*, overwrite: bool = False) -> None:
    """Register LoRA, QLoRA, and Full FT strategies."""
    from aiodoo_training.registries import adaptation_registry

    mappings: dict[str, type[AdaptationStrategy]] = {
        "lora": LoraAdaptationStrategy,
        "qlora": QLoraAdaptationStrategy,
        "full": FullFineTuneAdaptationStrategy,
        "full_finetune": FullFineTuneAdaptationStrategy,
    }
    for key, cls in mappings.items():
        if not adaptation_registry.exists(key) or overwrite:
            adaptation_registry.register(key, cls, overwrite=overwrite)
