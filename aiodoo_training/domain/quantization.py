"""
Quantization policy — AIODOO abstraction over 4-bit / 8-bit / floating precisions.

Named ``QuantizationPolicy`` to align with DevicePolicy / PrecisionPolicy /
MemoryPolicy. Infrastructure (bitsandbytes, etc.) maps this policy; domain never
imports those libraries.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiodoo_training.domain.enums import Precision
from aiodoo_training.domain.resources import PrecisionPolicy


@dataclass(frozen=True, slots=True)
class QuantizationPolicy:
    """
    Declared load-time quantization / precision policy.

    ``compute`` covers FP32 / FP16 / BF16 / INT8 / INT4 preferences.
    Boolean flags request quantized weight loading without naming a library.
    """

    compute: Precision = Precision.BF16
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    def __post_init__(self) -> None:
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("QuantizationPolicy cannot set both load_in_4bit and load_in_8bit.")

    @classmethod
    def from_precision_policy(cls, policy: PrecisionPolicy) -> QuantizationPolicy:
        """Map a frozen PrecisionPolicy into a QuantizationPolicy."""
        return cls(
            compute=policy.compute,
            load_in_4bit=policy.load_in_4bit,
            load_in_8bit=policy.load_in_8bit,
        )

    def canonical_parts(self) -> tuple[str, ...]:
        """Stable parts for fingerprinting."""
        return (
            f"compute={self.compute.value}",
            f"4bit={self.load_in_4bit}",
            f"8bit={self.load_in_8bit}",
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable mapping."""
        return {
            "compute": self.compute.value,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> QuantizationPolicy:
        """Rehydrate from :meth:`to_dict` output."""
        return cls(
            compute=Precision(str(data.get("compute", "bf16"))),
            load_in_4bit=bool(data.get("load_in_4bit", False)),
            load_in_8bit=bool(data.get("load_in_8bit", False)),
        )


# Backward-compatible alias (Phase 2 hardening — prefer QuantizationPolicy).
QuantizationSpec = QuantizationPolicy
