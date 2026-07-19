"""Resource management architecture tests (CPU-only)."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aiodoo_training.bootstrap import bootstrap_phase1
from aiodoo_training.domain import (
    AcceleratorKind,
    DeviceKind,
    DevicePolicy,
    ExecutionEnvironment,
    ExecutionSpec,
    HardwareCapabilities,
    MemoryPolicy,
    Precision,
    PrecisionPolicy,
)
from aiodoo_training.exceptions import DomainError
from aiodoo_training.factories import ResourcePlannerFactory
from aiodoo_training.infrastructure.resources import StaticResourcePlanner, TorchResourcePlanner
from aiodoo_training.ports.model import AdaptationStrategy, ModelBackend
from aiodoo_training.ports.resources import ResourcePlanner


def test_resource_domain_objects_are_immutable() -> None:
    policy = DevicePolicy()
    with pytest.raises(FrozenInstanceError):
        policy.preferred = DeviceKind.CUDA  # type: ignore[misc]
    caps = HardwareCapabilities()
    with pytest.raises(FrozenInstanceError):
        caps.device_count = 1  # type: ignore[misc]


def test_static_planner_resolves_cpu_and_falls_back() -> None:
    planner = StaticResourcePlanner()
    env = planner.resolve(
        DevicePolicy(preferred=DeviceKind.CUDA, allow_cpu_fallback=True),
        PrecisionPolicy(compute=Precision.BF16),
        MemoryPolicy(),
        accelerator=AcceleratorKind.FSDP,
    )
    assert isinstance(env, ExecutionEnvironment)
    assert env.selected_device == DeviceKind.CPU
    assert env.accelerator == AcceleratorKind.NONE
    assert "fallback" in env.metadata


def test_static_planner_rejects_unavailable_without_fallback() -> None:
    planner = StaticResourcePlanner()
    with pytest.raises(DomainError, match="allow_cpu_fallback"):
        planner.resolve(
            DevicePolicy(preferred=DeviceKind.CUDA, allow_cpu_fallback=False),
            PrecisionPolicy(),
            MemoryPolicy(),
        )


def test_resource_planner_factory_creates_static() -> None:
    bootstrap_phase1(overwrite=True)
    planner = ResourcePlannerFactory().create("static")
    assert isinstance(planner, ResourcePlanner)
    env = planner.resolve_spec(ExecutionSpec())
    assert env.selected_device == DeviceKind.CPU


def test_resource_planner_factory_creates_torch() -> None:
    bootstrap_phase1(overwrite=True)
    planner = ResourcePlannerFactory().create("torch")
    assert isinstance(planner, TorchResourcePlanner)


def test_torch_planner_auto_selects_cuda_when_available() -> None:
    planner = TorchResourcePlanner()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 1
    fake_torch.cuda.get_device_name.return_value = "Tesla T4"
    fake_torch.cuda.get_device_properties.return_value = SimpleNamespace(total_memory=15 * 1024**3)
    fake_torch.cuda.get_device_capability.return_value = (7, 5)
    fake_torch.cuda.is_bf16_supported.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        env = planner.resolve(
            DevicePolicy(preferred=DeviceKind.AUTO, allow_cpu_fallback=False),
            PrecisionPolicy(compute=Precision.FP16),
            MemoryPolicy(),
        )

    assert env.selected_device == DeviceKind.CUDA
    assert DeviceKind.CUDA in env.capabilities.available_devices
    assert env.capabilities.supports_fp16 is True
    assert env.capabilities.supports_bf16 is False
    assert env.metadata.get("gpu_name") == "Tesla T4"


def test_torch_planner_auto_selects_cpu_without_cuda() -> None:
    planner = TorchResourcePlanner()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        env = planner.resolve(
            DevicePolicy(preferred=DeviceKind.AUTO, allow_cpu_fallback=False),
            PrecisionPolicy(),
            MemoryPolicy(),
        )

    assert env.selected_device == DeviceKind.CPU


def test_torch_planner_rejects_cuda_without_fallback() -> None:
    planner = TorchResourcePlanner()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        with pytest.raises(DomainError, match="allow_cpu_fallback"):
            planner.resolve(
                DevicePolicy(preferred=DeviceKind.CUDA, allow_cpu_fallback=False),
                PrecisionPolicy(),
                MemoryPolicy(),
            )


def test_model_ports_require_execution_environment() -> None:
    """Public model ports must accept ExecutionEnvironment, not framework types."""
    assert "execution" in ModelBackend.load.__code__.co_varnames
    assert "execution" in AdaptationStrategy.apply.__code__.co_varnames
