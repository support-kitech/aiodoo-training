"""Scheduler port — opaque LR scheduler construction."""

from abc import ABC, abstractmethod

from aiodoo_training.domain.opt_handles import OptimizerHandle, SchedulerHandle
from aiodoo_training.domain.training_policies import SchedulerPolicy


class SchedulerBackend(ABC):
    """Builds an opaque LR scheduler handle."""

    @abstractmethod
    def build(self, optimizer: OptimizerHandle, policy: SchedulerPolicy) -> SchedulerHandle:
        """Construct scheduler for ``optimizer``."""
