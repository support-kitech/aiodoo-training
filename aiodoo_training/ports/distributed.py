"""Phase 7 distributed ports — framework-independent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from aiodoo_training.domain.config import DistributedSpec
from aiodoo_training.domain.device_mesh import DeviceMesh, PlacementPlan
from aiodoo_training.domain.distributed_session import DistributedTopology
from aiodoo_training.domain.enums import ReductionOp
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.domain.session import DatasetSession


class DistributedBackend(ABC):
    """Infrastructure adapter for process-group collectives."""

    @abstractmethod
    def initialize(self, topology: DistributedTopology) -> Mapping[str, str]:
        """Create process groups; return opaque group_id → status map."""

    @abstractmethod
    def barrier(self, group_id: str, *, timeout_sec: float) -> None:
        """Block until all ranks in the group arrive (or timeout)."""

    @abstractmethod
    def broadcast_bytes(self, group_id: str, payload: bytes, *, src_rank: int) -> bytes:
        """Broadcast bytes from src_rank; all ranks receive identical payload."""

    @abstractmethod
    def all_reduce_metrics(
        self,
        group_id: str,
        values: Mapping[str, float],
        *,
        op: ReductionOp,
    ) -> Mapping[str, float]:
        """Reduce metric values across ranks; return identical mapping on all ranks."""

    @abstractmethod
    def finalize(self) -> None:
        """Tear down process groups."""


class PlacementStrategy(ABC):
    """Registry-driven placement of ranks onto devices / mesh coordinates."""

    @abstractmethod
    def place(
        self,
        env: ExecutionEnvironment,
        topology: DistributedTopology,
        *,
        mesh_spec: DeviceMesh | None = None,
    ) -> PlacementPlan:
        """Produce an immutable PlacementPlan."""


class DistributedSampler(ABC):
    """Deterministic per-rank example index selection."""

    @abstractmethod
    def sample_indices(
        self,
        total: int,
        session: DatasetSession,
        seed: int,
    ) -> Sequence[int]:
        """Return ordered indices belonging to this rank/shard."""


class DistributedPlacementPort(ABC):
    """Optional port facade used by factories (implementation may be concrete app)."""

    @abstractmethod
    def resolve(
        self,
        env: ExecutionEnvironment,
        distributed: DistributedSpec,
        *,
        placement_key: str,
        mesh_axes: Sequence[str],
        mesh_shape: Sequence[int],
        communication_backend_key: str,
        runtime_backend_key: str,
    ) -> tuple[DeviceMesh, PlacementPlan, str]:
        """Return mesh, plan, and portable mesh_digest."""
