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
        worker_map: dict[int, WorkerStatus] = {}
        for worker_key, worker_status in self.workers.items():
            parsed_worker: WorkerStatus = (
                worker_status
                if isinstance(worker_status, WorkerStatus)
                else WorkerStatus(str(worker_status))
            )
            worker_map[int(worker_key)] = parsed_worker
        node_map: dict[str, NodeStatus] = {}
        for node_key, node_status in self.nodes.items():
            parsed_node: NodeStatus = (
                node_status if isinstance(node_status, NodeStatus) else NodeStatus(str(node_status))
            )
            node_map[str(node_key)] = parsed_node
        object.__setattr__(self, "workers", MappingProxyType(worker_map))
        object.__setattr__(self, "nodes", MappingProxyType(node_map))
