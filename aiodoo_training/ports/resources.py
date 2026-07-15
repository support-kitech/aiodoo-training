"""Resource planning port — probe hardware and resolve ExecutionEnvironment."""

from abc import ABC, abstractmethod

from aiodoo_training.domain.config import ExecutionSpec
from aiodoo_training.domain.enums import AcceleratorKind
from aiodoo_training.domain.resources import (
    DevicePolicy,
    ExecutionEnvironment,
    HardwareCapabilities,
    MemoryPolicy,
    PrecisionPolicy,
)


class ResourcePlanner(ABC):
    """
    Central entry point for hardware decisions.

    Application and pipeline stages must ask a ResourcePlanner instead of
    inspecting CUDA / MPS / XPU directly.
    """

    @abstractmethod
    def probe(self) -> HardwareCapabilities:
        """Discover available devices and precision support."""

    @abstractmethod
    def resolve(
        self,
        device: DevicePolicy,
        precision: PrecisionPolicy,
        memory: MemoryPolicy,
        *,
        accelerator: AcceleratorKind = AcceleratorKind.NONE,
    ) -> ExecutionEnvironment:
        """Resolve policies against probed capabilities into an environment."""

    def resolve_spec(self, spec: ExecutionSpec) -> ExecutionEnvironment:
        """Convenience: resolve a configured ExecutionSpec."""
        return self.resolve(
            spec.device,
            spec.precision,
            spec.memory,
            accelerator=spec.accelerator,
        )
