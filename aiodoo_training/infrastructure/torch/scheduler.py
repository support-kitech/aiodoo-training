"""Stub LR scheduler backends — opaque dict handles, no Torch required."""

from __future__ import annotations

from typing import Any

from aiodoo_training.domain.opt_handles import OptimizerHandle, SchedulerHandle
from aiodoo_training.domain.training_policies import SchedulerPolicy
from aiodoo_training.exceptions import DomainError
from aiodoo_training.ports.scheduler import SchedulerBackend


class StubSchedulerBackend(SchedulerBackend):
    """
    CPU stub scheduler for ``cosine`` / ``linear`` / ``constant``.

    Returns an opaque dict; real LR math is applied by Torch trainers later.
    """

    def build(self, optimizer: OptimizerHandle, policy: SchedulerPolicy) -> SchedulerHandle:
        name = policy.name.strip().lower()
        if name not in {"cosine", "linear", "constant"}:
            raise DomainError(f"Unsupported stub scheduler: {policy.name!r}.")
        opt_payload = optimizer if isinstance(optimizer, dict) else {"raw": optimizer}
        handle: dict[str, Any] = {
            "kind": name,
            "warmup_ratio": policy.warmup_ratio,
            "total_steps": policy.total_steps,
            "optimizer": opt_payload,
            "last_epoch": -1,
            "state": {},
        }
        return SchedulerHandle(handle)


def register_default_schedulers(*, overwrite: bool = False) -> None:
    """Register cosine / linear / constant schedulers (same stub class)."""
    from aiodoo_training.registries import scheduler_registry

    for key in ("cosine", "linear", "constant"):
        if not scheduler_registry.exists(key) or overwrite:
            scheduler_registry.register(key, StubSchedulerBackend, overwrite=overwrite)
