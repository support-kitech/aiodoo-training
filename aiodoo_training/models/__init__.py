"""Model loading package (Phase 2) — orchestration only; no framework imports."""

from aiodoo_training.models.access import (
    fingerprint_from_base_handle,
    metadata_from_base_handle,
    metadata_from_trainable_handle,
)
from aiodoo_training.models.fingerprints import (
    combine_model_adaptation_digests,
    fingerprint_adapter,
    fingerprint_execution,
    fingerprint_model,
    fingerprint_quantization,
)
from aiodoo_training.models.loader import LoadedModelContext, ModelLoader
from aiodoo_training.models.profiles import register_default_model_profiles

__all__ = [
    "LoadedModelContext",
    "ModelLoader",
    "combine_model_adaptation_digests",
    "fingerprint_adapter",
    "fingerprint_execution",
    "fingerprint_from_base_handle",
    "fingerprint_model",
    "fingerprint_quantization",
    "metadata_from_base_handle",
    "metadata_from_trainable_handle",
    "register_default_model_profiles",
]
