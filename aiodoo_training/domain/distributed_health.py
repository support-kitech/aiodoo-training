"""Phase 7 distributed health domain DTOs (never TrackingHealth)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from aiodoo_training.domain.enums import ClusterStatus, NodeStatus, WorkerStatus


@dataclass(frozen=True, slots=True)
class DistributedHealth:
    """Immutable backend-mesh / cluster health snapshot (runtime only)."""

    cluster: ClusterStatus = ClusterStatus.UNKNOWN
    workers: Mapping[int, WorkerStatus] = field(default_factory=lambda: MappingProxyType({}))
    nodes: Mapping[str, NodeStatus] = field(default_factory=lambda: MappingProxyType({}))
    message: str | None = None
    consecutive_barrier_timeouts: int = 0

    def __post_init__(self) -> None:
        if self.consecutive_barrier_timeouts < 0:
            raise ValueError("consecutive_barrier_timeouts must be >= 0.")
        workers: dict[int, WorkerStatus] = {}
        for k, v in self.workers.items():
            workers[int(k)] = v if isinstance(v, WorkerStatus) else WorkerStatus(str(v))
        nodes: dict[str, NodeStatus] = {}
        for k, v in self.nodes.items():
            nodes[str(k)] = v if isinstance(v, NodeStatus) else NodeStatus(str(v))
        object.__setattr__(self, "workers", MappingProxyType(workers))
        object.__setattr__(self, "nodes", MappingProxyType(nodes))
