"""Phase 7 distributed session / topology domain DTOs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType

from aiodoo_training.domain.enums import DistributedStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId


@dataclass(frozen=True, slots=True)
class Node:
    """One host in the distributed topology."""

    node_id: str
    hostname: str | None = None
    local_ranks: tuple[int, ...] = ()
    device_ids: tuple[int, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.node_id or not self.node_id.strip():
            raise ValueError("Node.node_id must be non-empty.")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType({str(k): str(v) for k, v in self.metadata.items()}),
        )


@dataclass(frozen=True, slots=True)
class ProcessGroupHandle:
    """Opaque named process-group handle (never a framework object).

    Future versions may expose backend capabilities through a **companion**
    capability object (similar to Phase 6 ``TrackingCapability``) without
    changing this handle's fields or public API.
    """

    group_id: str
    rank_set: tuple[int, ...]
    backend_key: str = "fake"

    def __post_init__(self) -> None:
        if not self.group_id or not self.group_id.strip():
            raise ValueError("ProcessGroupHandle.group_id must be non-empty.")
        if not self.backend_key or not self.backend_key.strip():
            raise ValueError("ProcessGroupHandle.backend_key must be non-empty.")
        if not self.rank_set:
            raise ValueError("ProcessGroupHandle.rank_set must be non-empty.")


@dataclass(frozen=True, slots=True)
class DistributedTopology:
    """Immutable mesh of nodes × ranks + process groups."""

    world_size: int
    global_rank: int
    local_rank: int
    node_id: str
    nodes: tuple[Node, ...] = ()
    groups: Mapping[str, ProcessGroupHandle] = field(
        default_factory=lambda: MappingProxyType({})
    )
    mesh_digest: str = ""

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("DistributedTopology.world_size must be >= 1.")
        if self.global_rank < 0 or self.global_rank >= self.world_size:
            raise ValueError("DistributedTopology.global_rank out of range.")
        if self.local_rank < 0:
            raise ValueError("DistributedTopology.local_rank must be >= 0.")
        if not self.node_id or not self.node_id.strip():
            raise ValueError("DistributedTopology.node_id must be non-empty.")
        object.__setattr__(
            self,
            "groups",
            MappingProxyType({str(k): v for k, v in self.groups.items()}),
        )


@dataclass(frozen=True, slots=True)
class DistributedSession:
    """Immutable lifecycle cursor for one distributed job."""

    session_id: str
    topology: DistributedTopology
    runtime_backend_key: str = "fake"
    status: DistributedStatus = DistributedStatus.PENDING
    experiment_id: ExperimentId | None = None
    run_id: RunId | None = None
    started_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("DistributedSession.session_id must be non-empty.")
        if not self.runtime_backend_key or not self.runtime_backend_key.strip():
            raise ValueError("DistributedSession.runtime_backend_key must be non-empty.")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType({str(k): str(v) for k, v in self.metadata.items()}),
        )

    def with_status(self, status: DistributedStatus) -> DistributedSession:
        return replace(self, status=status, updated_at=datetime.now(UTC))
