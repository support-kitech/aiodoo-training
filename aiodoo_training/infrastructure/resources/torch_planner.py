"""Torch-backed ResourcePlanner for production GPU runs (lazy Torch import)."""

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


class TorchResourcePlanner(ResourcePlanner):
    """
    Production resource planner that probes CUDA via Torch.

    Keeps Torch imports inside ``probe`` so importing this module remains safe
    in CPU-only CI. ``StaticResourcePlanner`` remains the CI / CPU default.
    """

    PLANNER_KEY = "torch"

    def probe(self) -> HardwareCapabilities:
        try:
            import torch
        except ImportError:
            return HardwareCapabilities(
                available_devices=(DeviceKind.CPU,),
                device_count=0,
                supports_bf16=False,
                supports_fp16=False,
                supports_tf32=False,
                extra=MappingProxyType(
                    {
                        "planner": "torch",
                        "torch": "unavailable",
                    }
                ),
            )

        cuda_available = bool(torch.cuda.is_available())
        if not cuda_available:
            return HardwareCapabilities(
                available_devices=(DeviceKind.CPU,),
                device_count=0,
                supports_bf16=False,
                supports_fp16=False,
                supports_tf32=False,
                extra=MappingProxyType(
                    {
                        "planner": "torch",
                        "cuda_available": "false",
                    }
                ),
            )

        device_count = int(torch.cuda.device_count())
        gpu_name = ""
        total_memory_bytes = 0
        supports_bf16 = False
        supports_fp16 = True  # CUDA GPUs used for training support fp16 compute
        supports_tf32 = False

        if device_count > 0:
            gpu_name = str(torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            total_memory_bytes = int(getattr(props, "total_memory", 0) or 0)
            major, minor = torch.cuda.get_device_capability(0)
            # TF32 is Ampere (sm_80) and newer.
            supports_tf32 = major >= 8
            if hasattr(torch.cuda, "is_bf16_supported"):
                supports_bf16 = bool(torch.cuda.is_bf16_supported())
            else:
                # bf16 requires Ampere+ when the helper is absent.
                supports_bf16 = major >= 8
            _ = minor

        extra: dict[str, str] = {
            "planner": "torch",
            "cuda_available": "true",
            "gpu_name": gpu_name,
            "total_memory_bytes": str(total_memory_bytes),
            "total_memory_gb": (
                f"{total_memory_bytes / (1024**3):.2f}" if total_memory_bytes else "0"
            ),
        }

        return HardwareCapabilities(
            available_devices=(DeviceKind.CPU, DeviceKind.CUDA),
            device_count=device_count,
            supports_bf16=supports_bf16,
            supports_fp16=supports_fp16,
            supports_tf32=supports_tf32,
            extra=MappingProxyType(extra),
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
        notes: dict[str, str] = {"planner": "torch"}
        if device.preferred not in {DeviceKind.AUTO, DeviceKind.CPU} and selected == DeviceKind.CPU:
            notes["fallback"] = f"preferred={device.preferred.value} unavailable; selected=cpu"
        if accelerator != AcceleratorKind.NONE and selected == DeviceKind.CPU:
            notes["accelerator_deferred"] = (
                f"{accelerator.value} declared but selected device is cpu"
            )
        # Preserve GPU identity metadata for operators / fingerprints.
        for key in ("gpu_name", "total_memory_gb", "cuda_available"):
            if key in capabilities.extra:
                notes[key] = capabilities.extra[key]

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
        available = set(capabilities.available_devices)
        cuda_available = DeviceKind.CUDA in available

        if preferred == DeviceKind.AUTO:
            return DeviceKind.CUDA if cuda_available else DeviceKind.CPU

        if preferred == DeviceKind.CPU:
            return DeviceKind.CPU

        if preferred in available:
            return preferred

        if policy.allow_cpu_fallback:
            return DeviceKind.CPU

        raise DomainError(
            f"Preferred device '{preferred.value}' is unavailable and "
            "DevicePolicy.allow_cpu_fallback is false."
        )
