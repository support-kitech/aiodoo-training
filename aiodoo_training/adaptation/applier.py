"""Adaptation application orchestration (Phase 2) — no framework imports."""

from __future__ import annotations

from dataclasses import dataclass

from aiodoo_training.determinism.model_fingerprints import fingerprint_adapter
from aiodoo_training.domain.adapter_info import AdapterFingerprint, AdapterMetadata
from aiodoo_training.domain.config import AdaptationSpec
from aiodoo_training.domain.handles import BaseModelHandle, TrainableModelHandle
from aiodoo_training.domain.quantization import QuantizationPolicy
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.models.access import metadata_from_trainable_handle
from aiodoo_training.ports.model import AdaptationStrategy


@dataclass(frozen=True, slots=True)
class AdaptedModelContext:
    """Result of applying an AdaptationStrategy — AIODOO types only."""

    handle: TrainableModelHandle
    metadata: AdapterMetadata
    fingerprint: AdapterFingerprint
    trainable_parameters: int
    execution: ExecutionEnvironment


class AdaptationApplier:
    """Applies an AdaptationStrategy under a resolved ExecutionEnvironment."""

    def __init__(self, strategy: AdaptationStrategy) -> None:
        self._strategy = strategy

    def apply(
        self,
        base_model: BaseModelHandle,
        spec: AdaptationSpec,
        execution: ExecutionEnvironment,
        *,
        quantization: QuantizationPolicy | None = None,
    ) -> AdaptedModelContext:
        """Apply adaptation and return an immutable context."""
        handle = self._strategy.apply(base_model, spec, execution)
        metadata = metadata_from_trainable_handle(handle)
        quant = quantization or QuantizationPolicy.from_precision_policy(execution.precision_policy)
        fingerprint = fingerprint_adapter(spec, quantization=quant)
        count = self._strategy.trainable_parameter_count(handle)
        return AdaptedModelContext(
            handle=handle,
            metadata=metadata,
            fingerprint=fingerprint,
            trainable_parameters=count,
            execution=execution,
        )
