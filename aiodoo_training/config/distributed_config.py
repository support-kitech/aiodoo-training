"""Phase 7 distributed configuration fragments."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aiodoo_training.domain.config import DistributedSpec
from aiodoo_training.domain.distributed_policies import (
    BarrierPolicy,
    CollectivePolicy,
    CommunicationBackendSpec,
    DistributedCheckpointPolicy,
    DistributedRuntimePolicy,
    EvaluationMergePolicy,
    ExportWritePolicy,
    MergePolicy,
    ReplicaPolicy,
    RestartPolicy,
    ShardPolicy,
)
from aiodoo_training.domain.enums import (
    BarrierTimeoutAction,
    DistributedCheckpointMode,
    RestartFrom,
)
from aiodoo_training.exceptions import ConfigError


class DistCommunicationFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "fake"
    timeout_sec: float = Field(default=1800.0, gt=0)
    require_deterministic_order: bool = True


class DistShardFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    shard_by: str = "rank"


class DistMergeFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    merge_on_export: bool = True


class DistReplicaFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replica_count: int = Field(default=1, ge=1)


class DistCheckpointFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["rank0_full", "sharded", "hybrid"] = "rank0_full"
    coordinator_rank: int = Field(default=0, ge=0)
    shard: DistShardFragment = Field(default_factory=DistShardFragment)
    merge: DistMergeFragment = Field(default_factory=DistMergeFragment)
    replica: DistReplicaFragment = Field(default_factory=DistReplicaFragment)


class DistFaultToleranceFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_restarts: int = Field(default=0, ge=0)
    restart_from: Literal["last_ckpt", "scratch"] = "last_ckpt"
    require_same_world_size: bool = True
    require_same_mesh_digest: bool = True


class DistTopologyFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    placement: str = "single"
    mesh_axes: list[str] = Field(default_factory=lambda: ["data"])
    mesh_shape: list[int] = Field(default_factory=lambda: [1])


class Phase7DistributedFragment(BaseModel):
    """Additive Phase 7 fragment composed over frozen DistributedSpec fields."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    backend: str = "fake"
    world_size: int = Field(default=1, ge=1)
    num_nodes: int = Field(default=1, ge=1)
    global_rank: int = Field(default=0, ge=0)
    local_rank: int = Field(default=0, ge=0)
    communication: DistCommunicationFragment = Field(default_factory=DistCommunicationFragment)
    checkpoint: DistCheckpointFragment = Field(default_factory=DistCheckpointFragment)
    fault_tolerance: DistFaultToleranceFragment = Field(default_factory=DistFaultToleranceFragment)
    topology: DistTopologyFragment = Field(default_factory=DistTopologyFragment)

    @field_validator("backend")
    @classmethod
    def _backend_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("distributed.backend must be non-empty")
        return value


def parse_phase7_distributed_config(data: dict[str, Any] | None) -> Phase7DistributedFragment:
    try:
        return Phase7DistributedFragment.model_validate(data or {})
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid distributed config: {exc}") from exc


def to_distributed_spec(fragment: Phase7DistributedFragment) -> DistributedSpec:
    enabled = fragment.enabled
    world = fragment.world_size if enabled else 1
    return DistributedSpec(
        enabled=enabled,
        world_size=world,
        global_rank=fragment.global_rank if enabled else 0,
        local_rank=fragment.local_rank if enabled else 0,
        num_nodes=fragment.num_nodes if enabled else 1,
    )


def to_runtime_policy(fragment: Phase7DistributedFragment) -> DistributedRuntimePolicy:
    product = 1
    for dim in fragment.topology.mesh_shape:
        product *= int(dim)
    if fragment.enabled and product != fragment.world_size:
        raise ConfigError(
            f"topology.mesh_shape product ({product}) must equal world_size ({fragment.world_size})"
        )
    if not fragment.enabled:
        return DistributedRuntimePolicy(enabled=False)

    mode = DistributedCheckpointMode(fragment.checkpoint.mode)
    return DistributedRuntimePolicy(
        enabled=True,
        backend_key=fragment.backend,
        placement_key=fragment.topology.placement,
        communication=CommunicationBackendSpec(
            key=fragment.communication.backend,
            timeout_sec=fragment.communication.timeout_sec,
        ),
        collective=CollectivePolicy(
            require_deterministic_order=fragment.communication.require_deterministic_order,
        ),
        barrier=BarrierPolicy(
            timeout_sec=fragment.communication.timeout_sec,
            on_timeout=BarrierTimeoutAction.FAIL,
        ),
        checkpoint=DistributedCheckpointPolicy(
            mode=mode,
            coordinator_rank=fragment.checkpoint.coordinator_rank,
            shard=ShardPolicy(
                enabled=fragment.checkpoint.shard.enabled,
                shard_by=fragment.checkpoint.shard.shard_by,
            ),
            merge=MergePolicy(
                enabled=fragment.checkpoint.merge.enabled,
                merge_on_export=fragment.checkpoint.merge.merge_on_export,
            ),
            replica=ReplicaPolicy(
                replica_count=fragment.checkpoint.replica.replica_count,
            ),
        ),
        restart=RestartPolicy(
            max_restarts=fragment.fault_tolerance.max_restarts,
            restart_from=RestartFrom(fragment.fault_tolerance.restart_from),
            require_same_world_size=fragment.fault_tolerance.require_same_world_size,
            require_same_mesh_digest=fragment.fault_tolerance.require_same_mesh_digest,
        ),
        export_write=ExportWritePolicy(),
        evaluation_merge=EvaluationMergePolicy(),
        mesh_axes=tuple(fragment.topology.mesh_axes),
        mesh_shape=tuple(int(s) for s in fragment.topology.mesh_shape),
    )


def validate_phase7_distributed_fragments(fragment: Phase7DistributedFragment) -> None:
    if fragment.enabled and fragment.world_size < 1:
        raise ConfigError("distributed.world_size must be >= 1")
    if fragment.checkpoint.coordinator_rank >= fragment.world_size and fragment.enabled:
        raise ConfigError("checkpoint.coordinator_rank must be < world_size")
    _ = to_runtime_policy(fragment)
