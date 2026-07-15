"""Additive optimizer port — opaque optimizer construction."""

from abc import ABC, abstractmethod

from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.opt_handles import OptimizerHandle
from aiodoo_training.domain.training_policies import OptimizerPolicy

# Backward-compatible re-exports after port split.
from aiodoo_training.ports.callback import TrainingCallback  # noqa: F401
from aiodoo_training.ports.scheduler import SchedulerBackend  # noqa: F401

__all__ = [
    "OptimizerBackend",
    "SchedulerBackend",
    "TrainingCallback",
]


class OptimizerBackend(ABC):
    """Builds an opaque optimizer handle for a trainable model."""

    @abstractmethod
    def build(self, model: TrainableModelHandle, policy: OptimizerPolicy) -> OptimizerHandle:
        """Construct optimizer state for ``model``."""
