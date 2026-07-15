"""
CPU-first resource planner (no Torch / CUDA imports).

Provides a stable ExecutionEnvironment for Phase 1+ until GPU probes land
under a Torch-backed planner in infrastructure.
"""

from __future__ import annotations

from types import MappingProxyType

from aiodoo_training.domain.enums import AcceleratorKind, DeviceKind
from aiodoo_training.domain.resources import (
    DevicePolicy,
    ExecutionEnvironment,
    HardwareCapabilities,
    MemoryPolicy,
    PrecisionPolicy,
)
from aiodoo_training.exceptions import DomainError
from aiodoo_training.ports.resources import ResourcePlanner


class StaticResourcePlanner(ResourcePlanner):
    """
    Resource planner that only advertises CPU.

    Preferred GPU / MPS / XPU devices fall back to CPU when
    ``DevicePolicy.allow_cpu_fallback`` is true; otherwise a DomainError is raised.
    """

    def probe(self) -> HardwareCapabilities:
        return HardwareCapabilities(
            available_devices=(DeviceKind.CPU,),
            device_count=0,
            supports_bf16=False,
            supports_fp16=False,
            supports_tf32=False,
            extra=MappingProxyType({"planner": "static_cpu"}),
        )

    def resolve(
        self,
        device: DevicePolicy,
        precision: PrecisionPolicy,
        memory: MemoryPolicy,
        *,
        accelerator: AcceleratorKind = AcceleratorKind.NONE,
    ) -> ExecutionEnvironment:
        capabilities = self.probe()
        selected = self._select_device(device, capabilities)
        notes: dict[str, str] = {"planner": "static_cpu"}
        if device.preferred not in {DeviceKind.AUTO, DeviceKind.CPU} and selected == DeviceKind.CPU:
            notes["fallback"] = f"preferred={device.preferred.value} unavailable; selected=cpu"
        if accelerator != AcceleratorKind.NONE:
            notes["accelerator_deferred"] = (
                f"{accelerator.value} declared but not activated by StaticResourcePlanner"
            )
        return ExecutionEnvironment(
            device_policy=device,
            precision_policy=precision,
            memory_policy=memory,
            capabilities=capabilities,
            selected_device=selected,
            accelerator=(AcceleratorKind.NONE if selected == DeviceKind.CPU else accelerator),
            metadata=MappingProxyType(notes),
        )

    @staticmethod
    def _select_device(
        policy: DevicePolicy,
        capabilities: HardwareCapabilities,
    ) -> DeviceKind:
        preferred = policy.preferred
        if preferred in {DeviceKind.AUTO, DeviceKind.CPU}:
            return DeviceKind.CPU
        if preferred in capabilities.available_devices:
            return preferred
        if policy.allow_cpu_fallback:
            return DeviceKind.CPU
        raise DomainError(
            f"Preferred device '{preferred.value}' is unavailable and "
            "DevicePolicy.allow_cpu_fallback is false."
        )


def register_default_resource_planners(*, overwrite: bool = False) -> None:
    """Register the static CPU planner under the 'static' / 'cpu' keys."""
    from aiodoo_training.registries import resource_planner_registry

    for key in ("static", "cpu"):
        if not resource_planner_registry.exists(key) or overwrite:
            resource_planner_registry.register(
                key,
                StaticResourcePlanner,
                overwrite=overwrite,
            )
