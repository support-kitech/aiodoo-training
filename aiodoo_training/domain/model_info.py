"""Phase 2 model metadata — framework-independent, immutable, serializable."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from aiodoo_training.domain.enums import ModelFamily, Precision
from aiodoo_training.domain.quantization import QuantizationPolicy


def _freeze_str_map(value: Mapping[str, str] | None) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType({str(k): str(v) for k, v in value.items()})


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Declared capabilities of a base model profile or loaded model."""

    supports_gradient_checkpointing: bool = True
    supports_flash_attention: bool = False
    max_position_embeddings: int | None = None
    vocab_size: int | None = None
    hidden_size: int | None = None
    num_parameters: int | None = None
    extra: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_str_map(self.extra))

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable mapping."""
        return {
            "supports_gradient_checkpointing": self.supports_gradient_checkpointing,
            "supports_flash_attention": self.supports_flash_attention,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_parameters": self.num_parameters,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelCapabilities:
        """Rehydrate from :meth:`to_dict` output."""
        return cls(
            supports_gradient_checkpointing=bool(data.get("supports_gradient_checkpointing", True)),
            supports_flash_attention=bool(data.get("supports_flash_attention", False)),
            max_position_embeddings=(
                int(data["max_position_embeddings"])
                if data.get("max_position_embeddings") is not None
                else None
            ),
            vocab_size=int(data["vocab_size"]) if data.get("vocab_size") is not None else None,
            hidden_size=int(data["hidden_size"]) if data.get("hidden_size") is not None else None,
            num_parameters=(
                int(data["num_parameters"]) if data.get("num_parameters") is not None else None
            ),
            extra=_freeze_str_map(data.get("extra") or {}),
        )


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """
    Immutable description of a base model (no framework types).

    Serializable via :meth:`to_dict` / :meth:`from_dict` for fingerprints,
    manifests, and future checkpoint sidecars — never embeds Torch/HF objects.
    """

    identifier: str
    family: ModelFamily
    revision: str | None = None
    precision: Precision = Precision.BF16
    quantization: QuantizationPolicy = field(default_factory=QuantizationPolicy)
    tokenizer_binding: str | None = None
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    backend_key: str = "stub"
    extra: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.identifier or not self.identifier.strip():
            raise ValueError("ModelMetadata.identifier must be a non-empty string.")
        object.__setattr__(self, "extra", _freeze_str_map(self.extra))

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable mapping (framework-independent)."""
        return {
            "identifier": self.identifier,
            "family": self.family.value,
            "revision": self.revision,
            "precision": self.precision.value,
            "quantization": self.quantization.to_dict(),
            "tokenizer_binding": self.tokenizer_binding,
            "capabilities": self.capabilities.to_dict(),
            "backend_key": self.backend_key,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelMetadata:
        """Rehydrate from :meth:`to_dict` output."""
        quant_raw = data.get("quantization") or {}
        caps_raw = data.get("capabilities") or {}
        if not isinstance(quant_raw, Mapping):
            raise ValueError("ModelMetadata.quantization must be a mapping.")
        if not isinstance(caps_raw, Mapping):
            raise ValueError("ModelMetadata.capabilities must be a mapping.")
        return cls(
            identifier=str(data["identifier"]),
            family=ModelFamily(str(data.get("family", "unknown"))),
            revision=str(data["revision"]) if data.get("revision") is not None else None,
            precision=Precision(str(data.get("precision", "bf16"))),
            quantization=QuantizationPolicy.from_dict(dict(quant_raw)),
            tokenizer_binding=(
                str(data["tokenizer_binding"])
                if data.get("tokenizer_binding") is not None
                else None
            ),
            capabilities=ModelCapabilities.from_dict(caps_raw),
            backend_key=str(data.get("backend_key", "stub")),
            extra=_freeze_str_map(data.get("extra") or {}),
        )


@dataclass(frozen=True, slots=True)
class ModelFingerprint:
    """Deterministic fingerprint of base-model identity for experiment hashing."""

    digest: str
    identifier: str
    revision: str | None
    family: str
    quantization_digest: str
    execution_digest: str

    def __post_init__(self) -> None:
        if len(self.digest) < 16:
            raise ValueError("ModelFingerprint.digest must be a non-trivial digest.")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Registered model profile (family defaults + tokenizer binding)."""

    key: str
    family: ModelFamily
    default_identifier: str
    tokenizer_binding: str
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    default_precision: Precision = Precision.BF16

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("ModelProfile.key must be a non-empty string.")
