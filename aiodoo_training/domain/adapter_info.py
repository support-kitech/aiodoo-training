"""Phase 2 adapter metadata — framework-independent, immutable."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from aiodoo_training.domain.config import AdaptationSpec
from aiodoo_training.domain.enums import AdapterType


def _freeze_str_map(value: Mapping[str, str] | None) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType({str(k): str(v) for k, v in value.items()})


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    """Declared capabilities of an adapter profile / applied adaptation."""

    supports_merge: bool = True
    supports_resume: bool = True
    requires_quantization: bool = False
    extra: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_str_map(self.extra))


@dataclass(frozen=True, slots=True)
class AdapterProfile:
    """
    Registered adapter profile — declarative metadata, not adaptation behavior.

    ``strategy_key`` selects an :class:`~aiodoo_training.ports.model.AdaptationStrategy`
    from ``adaptation_registry``. Profiles live in ``adapter_registry``.
    """

    key: str
    adapter_type: AdapterType
    strategy_key: str
    rank: int | None = 8
    alpha: int | None = 16
    dropout: float | None = 0.05
    target_modules: tuple[str, ...] = ()
    capabilities: AdapterCapabilities = field(default_factory=AdapterCapabilities)

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("AdapterProfile.key must be a non-empty string.")
        if not self.strategy_key or not self.strategy_key.strip():
            raise ValueError("AdapterProfile.strategy_key must be a non-empty string.")

    def to_adaptation_spec(self) -> AdaptationSpec:
        """Materialize an AdaptationSpec from this profile."""
        return AdaptationSpec(
            adapter_type=self.adapter_type,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=self.target_modules,
            extra=MappingProxyType({"profile": self.key}),
        )


@dataclass(frozen=True, slots=True)
class AdapterMetadata:
    """Immutable description of an applied adaptation (no PEFT types)."""

    adapter_type: AdapterType
    rank: int | None = None
    alpha: int | None = None
    dropout: float | None = None
    target_modules: tuple[str, ...] = ()
    trainable_parameters: int | None = None
    total_parameters: int | None = None
    capabilities: AdapterCapabilities = field(default_factory=AdapterCapabilities)
    strategy_key: str = "stub"
    profile_key: str | None = None
    extra: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_str_map(self.extra))

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable mapping."""
        return {
            "adapter_type": self.adapter_type.value,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "capabilities": {
                "supports_merge": self.capabilities.supports_merge,
                "supports_resume": self.capabilities.supports_resume,
                "requires_quantization": self.capabilities.requires_quantization,
                "extra": dict(self.capabilities.extra),
            },
            "strategy_key": self.strategy_key,
            "profile_key": self.profile_key,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True, slots=True)
class AdapterFingerprint:
    """Deterministic fingerprint of adaptation configuration."""

    digest: str
    adapter_type: str
    rank: int | None
    alpha: int | None
    target_modules: tuple[str, ...]
    quantization_digest: str

    def __post_init__(self) -> None:
        if len(self.digest) < 16:
            raise ValueError("AdapterFingerprint.digest must be a non-trivial digest.")
