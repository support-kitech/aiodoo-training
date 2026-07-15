"""Default model family / profile / capability registrations (Phase 2)."""

from __future__ import annotations

from aiodoo_training.domain.enums import ModelFamily, Precision
from aiodoo_training.domain.model_info import ModelCapabilities, ModelProfile
from aiodoo_training.registries import (
    model_capability_registry,
    model_family_registry,
    model_profile_registry,
)


def _default_profiles() -> tuple[ModelProfile, ...]:
    return (
        ModelProfile(
            key="qwen",
            family=ModelFamily.QWEN,
            default_identifier="Qwen/Qwen2.5-Coder-0.5B",
            tokenizer_binding="qwen",
            capabilities=ModelCapabilities(supports_gradient_checkpointing=True),
            default_precision=Precision.BF16,
        ),
        ModelProfile(
            key="llama",
            family=ModelFamily.LLAMA,
            default_identifier="meta-llama/Llama-3.2-1B",
            tokenizer_binding="llama",
            default_precision=Precision.BF16,
        ),
        ModelProfile(
            key="mistral",
            family=ModelFamily.MISTRAL,
            default_identifier="mistralai/Mistral-7B-v0.1",
            tokenizer_binding="mistral",
            default_precision=Precision.BF16,
        ),
        ModelProfile(
            key="deepseek",
            family=ModelFamily.DEEPSEEK,
            default_identifier="deepseek-ai/deepseek-coder-1.3b-base",
            tokenizer_binding="deepseek",
            default_precision=Precision.BF16,
        ),
        ModelProfile(
            key="gemma",
            family=ModelFamily.GEMMA,
            default_identifier="google/gemma-2b",
            tokenizer_binding="gemma",
            default_precision=Precision.BF16,
        ),
        ModelProfile(
            key="phi",
            family=ModelFamily.PHI,
            default_identifier="microsoft/phi-2",
            tokenizer_binding="phi",
            default_precision=Precision.BF16,
        ),
    )


def register_default_model_profiles(*, overwrite: bool = False) -> None:
    """Register model families, profiles, and capabilities (registration-driven)."""
    for profile in _default_profiles():
        family_key = profile.family.value
        if not model_family_registry.exists(family_key) or overwrite:
            model_family_registry.register(family_key, profile.family, overwrite=overwrite)
        if not model_profile_registry.exists(profile.key) or overwrite:
            model_profile_registry.register(profile.key, profile, overwrite=overwrite)
        if not model_capability_registry.exists(profile.key) or overwrite:
            model_capability_registry.register(
                profile.key,
                profile.capabilities,
                overwrite=overwrite,
            )
