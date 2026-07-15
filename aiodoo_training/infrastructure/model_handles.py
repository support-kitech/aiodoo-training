"""
Private opaque model containers for infrastructure backends.

Never imported outside ``aiodoo_training.infrastructure``. Attach only AIODOO
metadata attributes; callers use BaseModelHandle / TrainableModelHandle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiodoo_training.domain.adapter_info import AdapterMetadata
from aiodoo_training.domain.handles import BaseModelHandle, TrainableModelHandle
from aiodoo_training.domain.model_info import ModelFingerprint, ModelMetadata


@dataclass
class OpaqueBaseModel:
    """Carrier for a loaded base model (framework object stays private)."""

    framework_model: Any
    aiodoo_metadata: ModelMetadata
    aiodoo_fingerprint: ModelFingerprint
    backend_key: str


@dataclass
class OpaqueTrainableModel:
    """Carrier for an adapted trainable model (framework object stays private)."""

    framework_model: Any
    aiodoo_adapter_metadata: AdapterMetadata
    base: OpaqueBaseModel | None
    strategy_key: str


def as_base_handle(carrier: OpaqueBaseModel) -> BaseModelHandle:
    """Wrap a carrier as a BaseModelHandle (NewType identity at runtime)."""
    return BaseModelHandle(carrier)


def as_trainable_handle(carrier: OpaqueTrainableModel) -> TrainableModelHandle:
    """Wrap a carrier as a TrainableModelHandle."""
    return TrainableModelHandle(carrier)


def require_base_carrier(handle: BaseModelHandle) -> OpaqueBaseModel:
    """Unwrap a BaseModelHandle produced by infrastructure backends."""
    if not isinstance(handle, OpaqueBaseModel):
        raise TypeError(
            f"Expected OpaqueBaseModel inside BaseModelHandle, got {type(handle).__name__}."
        )
    return handle


def require_trainable_carrier(handle: TrainableModelHandle) -> OpaqueTrainableModel:
    """Unwrap a TrainableModelHandle produced by infrastructure backends."""
    if not isinstance(handle, OpaqueTrainableModel):
        raise TypeError(
            "Expected OpaqueTrainableModel inside TrainableModelHandle, "
            f"got {type(handle).__name__}."
        )
    return handle
