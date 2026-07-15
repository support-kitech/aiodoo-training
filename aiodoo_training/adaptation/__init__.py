"""Adaptation package (Phase 2) — orchestration only; PEFT stays in infrastructure."""

from aiodoo_training.adaptation.applier import AdaptationApplier, AdaptedModelContext
from aiodoo_training.adaptation.profiles import register_default_adapter_profiles

__all__ = [
    "AdaptationApplier",
    "AdaptedModelContext",
    "register_default_adapter_profiles",
]
