"""DistributedContext binder bag."""

from __future__ import annotations

from dataclasses import dataclass, replace

from aiodoo_training.domain.device_mesh import DeviceMesh, PlacementPlan
from aiodoo_training.domain.distributed_health import DistributedHealth
from aiodoo_training.domain.distributed_policies import DistributedRuntimePolicy
from aiodoo_training.domain.distributed_session import DistributedSession
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.ports.distributed import DistributedBackend


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Immutable binder bag for distributed runtime collaborators."""

    session: DistributedSession
    execution: ExecutionEnvironment
    policy: DistributedRuntimePolicy
    mesh: DeviceMesh
    placement: PlacementPlan
    backend: DistributedBackend
    health: DistributedHealth = DistributedHealth()

    def with_session(self, session: DistributedSession) -> DistributedContext:
        return replace(self, session=session)

    def with_health(self, health: DistributedHealth) -> DistributedContext:
        return replace(self, health=health)
