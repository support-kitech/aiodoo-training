"""
Resource management domain — centralizes future hardware decisions.

No Torch / CUDA / Accelerate imports belong here. Infrastructure probes and
maps these policies onto concrete runtimes in later phases.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from aiodoo_training.domain.enums import AcceleratorKind, DeviceKind, Precision


@dataclass(frozen=True, slots=True)
class DevicePolicy:
    """Preferred device selection without framework checks."""

    preferred: DeviceKind = DeviceKind.AUTO
    allow_cpu_fallback: bool = True
    device_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """
    Load / compute precision preference.

    Distinct from training-loop ``PrecisionSpec``: this policy governs how
    weights are placed on a device before adaptation/training.
    """

    compute: Precision = Precision.BF16
    load_in_4bit: bool = False
    load_in_8bit: bool = False


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    """Memory and offload preferences (framework-independent)."""

    max_memory_gb: float | None = None
    activation_checkpointing: bool = False
    allow_cpu_offload: bool = False


@dataclass(frozen=True, slots=True)
class HardwareCapabilities:
    """
    Discovered hardware capabilities.

    Produced by a ResourcePlanner probe — never by ad-hoc CUDA checks in
    domain or application code.
    """

    available_devices: tuple[DeviceKind, ...] = (DeviceKind.CPU,)
    device_count: int = 0
    supports_bf16: bool = False
    supports_fp16: bool = False
    supports_tf32: bool = False
    extra: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.device_count < 0:
            raise ValueError("HardwareCapabilities.device_count must be >= 0.")


@dataclass(frozen=True, slots=True)
class ExecutionEnvironment:
    """
    Resolved execution plan for a run.

    Later phases consult this object instead of calling ``torch.cuda`` or
    inspecting ``device_map`` directly from training code.
    """

    device_policy: DevicePolicy
    precision_policy: PrecisionPolicy
    memory_policy: MemoryPolicy
    capabilities: HardwareCapabilities
    selected_device: DeviceKind = DeviceKind.CPU
    accelerator: AcceleratorKind = AcceleratorKind.NONE
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
