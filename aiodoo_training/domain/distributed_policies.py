"""Phase 7 distributed policy domain DTOs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from aiodoo_training.domain.enums import (
    BarrierTimeoutAction,
    DistributedCheckpointMode,
    ReductionOp,
    RestartFrom,
)


@dataclass(frozen=True, slots=True)
class BarrierPolicy:
    timeout_sec: float = 1800.0
    on_timeout: BarrierTimeoutAction = BarrierTimeoutAction.FAIL

    def __post_init__(self) -> None:
        if self.timeout_sec <= 0:
            raise ValueError("BarrierPolicy.timeout_sec must be > 0.")


@dataclass(frozen=True, slots=True)
class BroadcastPolicy:
    src_rank: int = 0
    max_bytes: int = 16 * 1024 * 1024
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if self.src_rank < 0:
            raise ValueError("BroadcastPolicy.src_rank must be >= 0.")
        if self.max_bytes < 1:
            raise ValueError("BroadcastPolicy.max_bytes must be >= 1.")


@dataclass(frozen=True, slots=True)
class ReductionPolicy:
    op: ReductionOp = ReductionOp.MEAN
    dtype_hint: str | None = "fp32"


@dataclass(frozen=True, slots=True)
class AggregationPolicy:
    metric_keys: tuple[str, ...] = ()
    reduction: ReductionPolicy = field(default_factory=ReductionPolicy)
    broadcast_result: bool = True


@dataclass(frozen=True, slots=True)
class CommunicationBackendSpec:
    key: str = "fake"
    timeout_sec: float = 1800.0
    options: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("CommunicationBackendSpec.key must be non-empty.")
        if self.timeout_sec <= 0:
            raise ValueError("CommunicationBackendSpec.timeout_sec must be > 0.")
        object.__setattr__(
            self,
            "options",
            MappingProxyType({str(k): str(v) for k, v in self.options.items()}),
        )


@dataclass(frozen=True, slots=True)
class CollectivePolicy:
    default_group: str = "default"
    allow_async: bool = False
    require_deterministic_order: bool = True
    reduce_dtype_hint: str | None = "fp32"


@dataclass(frozen=True, slots=True)
class ShardPolicy:
    enabled: bool = False
    shard_by: str = "rank"
    filename_pattern: str = "shard-{rank}.bin"


@dataclass(frozen=True, slots=True)
class MergePolicy:
    enabled: bool = False
    merge_on_export: bool = True
    merge_on_resume_load: bool = False


@dataclass(frozen=True, slots=True)
class ReplicaPolicy:
    replica_count: int = 1
    replica_roots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.replica_count < 1:
            raise ValueError("ReplicaPolicy.replica_count must be >= 1.")


@dataclass(frozen=True, slots=True)
class DistributedCheckpointPolicy:
    """Who/how of distributed saves — orthogonal to Phase 3 CheckpointPolicy."""

    mode: DistributedCheckpointMode = DistributedCheckpointMode.RANK0_FULL
    coordinator_rank: int = 0
    require_barrier_before_save: bool = True
    require_barrier_after_publish: bool = True
    shard: ShardPolicy = field(default_factory=ShardPolicy)
    merge: MergePolicy = field(default_factory=MergePolicy)
    replica: ReplicaPolicy = field(default_factory=ReplicaPolicy)

    def __post_init__(self) -> None:
        if self.coordinator_rank < 0:
            raise ValueError("DistributedCheckpointPolicy.coordinator_rank must be >= 0.")


@dataclass(frozen=True, slots=True)
class RestartPolicy:
    """Relaunch decisions; never softens ResumePolicy compatibility gates."""

    max_restarts: int = 0
    backoff_sec: float = 0.0
    restart_from: RestartFrom = RestartFrom.LAST_CKPT
    require_same_world_size: bool = True
    require_same_mesh_digest: bool = True

    def __post_init__(self) -> None:
        if self.max_restarts < 0:
            raise ValueError("RestartPolicy.max_restarts must be >= 0.")
        if self.backoff_sec < 0:
            raise ValueError("RestartPolicy.backoff_sec must be >= 0.")


@dataclass(frozen=True, slots=True)
class ExportWritePolicy:
    writer_rank: int = 0
    require_barrier_before_export: bool = True
    require_merged_weights_if_sharded: bool = True

    def __post_init__(self) -> None:
        if self.writer_rank < 0:
            raise ValueError("ExportWritePolicy.writer_rank must be >= 0.")


@dataclass(frozen=True, slots=True)
class EvaluationMergePolicy:
    metric_ops: Mapping[str, ReductionOp] = field(default_factory=lambda: MappingProxyType({}))
    require_all_ranks: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metric_ops",
            MappingProxyType(
                {
                    str(k): (v if isinstance(v, ReductionOp) else ReductionOp(str(v)))
                    for k, v in self.metric_ops.items()
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class DistributedRuntimePolicy:
    """Resolved Phase 7 distributed preferences for a run."""

    enabled: bool = False
    backend_key: str = "fake"
    placement_key: str = "single"
    communication: CommunicationBackendSpec = field(default_factory=CommunicationBackendSpec)
    collective: CollectivePolicy = field(default_factory=CollectivePolicy)
    barrier: BarrierPolicy = field(default_factory=BarrierPolicy)
    checkpoint: DistributedCheckpointPolicy = field(default_factory=DistributedCheckpointPolicy)
    restart: RestartPolicy = field(default_factory=RestartPolicy)
    export_write: ExportWritePolicy = field(default_factory=ExportWritePolicy)
    evaluation_merge: EvaluationMergePolicy = field(default_factory=EvaluationMergePolicy)
    mesh_axes: tuple[str, ...] = ("data",)
    mesh_shape: tuple[int, ...] = (1,)
