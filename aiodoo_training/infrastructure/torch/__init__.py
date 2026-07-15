"""Torch-backed infrastructure adapters (RNG, optimizer, scheduler)."""

from aiodoo_training.infrastructure.torch.optimizer import (
    StubAdamWOptimizerBackend,
    register_default_optimizers,
)
from aiodoo_training.infrastructure.torch.rng import (
    PythonRngController,
    TorchRngController,
    register_default_rng,
)
from aiodoo_training.infrastructure.torch.scheduler import (
    StubSchedulerBackend,
    register_default_schedulers,
)

__all__ = [
    "PythonRngController",
    "StubAdamWOptimizerBackend",
    "StubSchedulerBackend",
    "TorchRngController",
    "register_default_optimizers",
    "register_default_rng",
    "register_default_schedulers",
]
