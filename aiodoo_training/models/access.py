"""
Opaque handle helpers (duck-typed AIODOO attributes only).

These helpers never import Torch / Transformers / PEFT. Backends attach
``aiodoo_metadata`` / ``aiodoo_adapter_metadata`` on opaque handle objects.
"""

from __future__ import annotations

from aiodoo_training.domain.adapter_info import AdapterMetadata
from aiodoo_training.domain.handles import BaseModelHandle, TrainableModelHandle
from aiodoo_training.domain.model_info import ModelFingerprint, ModelMetadata
from aiodoo_training.exceptions import DomainError


def metadata_from_base_handle(handle: BaseModelHandle) -> ModelMetadata:
    """Extract ModelMetadata attached by a ModelBackend implementation."""
    meta = getattr(handle, "aiodoo_metadata", None)
    if not isinstance(meta, ModelMetadata):
        raise DomainError(
            "BaseModelHandle is missing aiodoo_metadata; backend must attach ModelMetadata."
        )
    return meta


def fingerprint_from_base_handle(handle: BaseModelHandle) -> ModelFingerprint | None:
    """Extract optional ModelFingerprint attached by a ModelBackend."""
    fp = getattr(handle, "aiodoo_fingerprint", None)
    return fp if isinstance(fp, ModelFingerprint) else None


def metadata_from_trainable_handle(handle: TrainableModelHandle) -> AdapterMetadata:
    """Extract AdapterMetadata attached by an AdaptationStrategy implementation."""
    meta = getattr(handle, "aiodoo_adapter_metadata", None)
    if not isinstance(meta, AdapterMetadata):
        raise DomainError(
            "TrainableModelHandle is missing aiodoo_adapter_metadata; "
            "strategy must attach AdapterMetadata."
        )
    return meta
