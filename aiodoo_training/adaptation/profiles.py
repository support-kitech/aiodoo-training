"""Default adapter profiles — metadata independent of AdaptationStrategy classes."""

from __future__ import annotations

from aiodoo_training.domain.adapter_info import AdapterCapabilities, AdapterProfile
from aiodoo_training.domain.enums import AdapterType
from aiodoo_training.registries import adapter_registry


def _default_profiles() -> tuple[AdapterProfile, ...]:
    return (
        AdapterProfile(
            key="lora-r8",
            adapter_type=AdapterType.LORA,
            strategy_key="lora",
            rank=8,
            alpha=16,
            dropout=0.05,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            capabilities=AdapterCapabilities(supports_merge=True),
        ),
        AdapterProfile(
            key="qlora-r8",
            adapter_type=AdapterType.QLORA,
            strategy_key="qlora",
            rank=8,
            alpha=16,
            dropout=0.05,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            capabilities=AdapterCapabilities(
                supports_merge=True,
                requires_quantization=True,
            ),
        ),
        AdapterProfile(
            key="full",
            adapter_type=AdapterType.FULL,
            strategy_key="full",
            rank=None,
            alpha=None,
            dropout=None,
            target_modules=(),
            capabilities=AdapterCapabilities(supports_merge=False),
        ),
    )


def register_default_adapter_profiles(*, overwrite: bool = False) -> None:
    """Register declarative adapter profiles into adapter_registry."""
    for profile in _default_profiles():
        if not adapter_registry.exists(profile.key) or overwrite:
            adapter_registry.register(profile.key, profile, overwrite=overwrite)
