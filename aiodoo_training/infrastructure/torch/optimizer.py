"""Stub optimizer backends — opaque dict handles, no Torch required."""

from __future__ import annotations

from typing import Any

from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.opt_handles import OptimizerHandle
from aiodoo_training.domain.training_policies import OptimizerPolicy
from aiodoo_training.exceptions import DomainError
from aiodoo_training.ports.optimizer import OptimizerBackend


class StubAdamWOptimizerBackend(OptimizerBackend):
    """CPU stub AdamW — returns an opaque dict state handle."""

    BACKEND_KEY = "adamw"

    def build(self, model: TrainableModelHandle, policy: OptimizerPolicy) -> OptimizerHandle:
        if policy.name.strip().lower() not in {"adamw", "stub_adamw"}:
            raise DomainError(
                f"StubAdamWOptimizerBackend expects policy.name='adamw'; got {policy.name!r}."
            )
        _ = model  # Param groups would be derived from the carrier in a Torch impl.
        handle: dict[str, Any] = {
            "kind": "adamw",
            "learning_rate": policy.learning_rate,
            "weight_decay": policy.weight_decay,
            "beta1": policy.beta1,
            "beta2": policy.beta2,
            "eps": policy.eps,
            "step": 0,
            "state": {},
        }
        return OptimizerHandle(handle)


class StubOptimizerBackend(StubAdamWOptimizerBackend):
    """Alias kept for registration clarity."""


def register_default_optimizers(*, overwrite: bool = False) -> None:
    """Register stub AdamW under the ``adamw`` key."""
    from aiodoo_training.registries import optimizer_registry

    if not optimizer_registry.exists("adamw") or overwrite:
        optimizer_registry.register("adamw", StubAdamWOptimizerBackend, overwrite=overwrite)
