"""Distributed coordinators — never Authorities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from aiodoo_training.distributed.context import DistributedContext
from aiodoo_training.distributed.sync import SyncFacade
from aiodoo_training.domain.distributed_policies import (
    BarrierPolicy,
    DistributedCheckpointPolicy,
    EvaluationMergePolicy,
    ExportWritePolicy,
)
from aiodoo_training.domain.enums import (
    DistributedCheckpointMode,
    RankRole,
    ReductionOp,
)
from aiodoo_training.exceptions import DistributedError


class DistributedCoordinator:
    """Rank role assignment for distributed wiring."""

    def role_for(self, context: DistributedContext, concern: str) -> RankRole:
        rank = context.session.topology.global_rank
        policy = context.policy
        if concern in {"export", "eval_publish", "tracking_log"}:
            writer = policy.export_write.writer_rank
            return RankRole.COORDINATOR if rank == writer else RankRole.IDLE
        if concern == "checkpoint":
            return self._checkpoint_role(context, policy.checkpoint)
        return RankRole.IDLE

    def _checkpoint_role(
        self, context: DistributedContext, policy: DistributedCheckpointPolicy
    ) -> RankRole:
        rank = context.session.topology.global_rank
        if policy.mode is DistributedCheckpointMode.RANK0_FULL:
            return RankRole.COORDINATOR if rank == policy.coordinator_rank else RankRole.IDLE
        if policy.mode is DistributedCheckpointMode.SHARDED:
            if rank == policy.coordinator_rank:
                return RankRole.COORDINATOR
            return RankRole.SHARD_WRITER
        # HYBRID: coordinator + shard writers
        if rank == policy.coordinator_rank:
            return RankRole.COORDINATOR
        return RankRole.SHARD_WRITER


class DistributedCheckpointCoordinator:
    """
    Coordinates who may call CheckpointManager.

    Never writes weight packages itself. Never replaces CheckpointManager /
    ResumePolicy.
    """

    def __init__(
        self,
        sync: SyncFacade,
        roles: DistributedCoordinator | None = None,
    ) -> None:
        self._sync = sync
        self._roles = roles or DistributedCoordinator()

    def prepare_save(self, context: DistributedContext) -> RankRole:
        policy = context.policy.checkpoint
        if policy.require_barrier_before_save:
            self._sync.barrier(
                "ckpt_before",
                policy=context.policy.barrier,
            )
        return self._roles.role_for(context, "checkpoint")

    def finish_save(self, context: DistributedContext) -> None:
        if context.policy.checkpoint.require_barrier_after_publish:
            self._sync.barrier(
                "ckpt_after",
                policy=context.policy.barrier,
            )

    def may_invoke_checkpoint_manager(self, role: RankRole) -> bool:
        return role in {RankRole.COORDINATOR, RankRole.SHARD_WRITER}


class DistributedEvaluationCoordinator:
    """Shard eval + merge around EvaluationEngine (engine unchanged)."""

    def __init__(self, sync: SyncFacade) -> None:
        self._sync = sync

    def merge_metrics(
        self,
        local: Mapping[str, float],
        policy: EvaluationMergePolicy | None = None,
    ) -> Mapping[str, float]:
        pol = policy or EvaluationMergePolicy()
        if pol.require_all_ranks:
            self._sync.barrier("eval_merge")
        # Default: mean all local values; honor per-key ops when provided.
        if not pol.metric_ops:
            return self._sync.all_reduce(local, op=ReductionOp.MEAN)
        out: dict[str, float] = {}
        for key, value in local.items():
            op = pol.metric_ops.get(key, ReductionOp.MEAN)
            reduced = self._sync.all_reduce({key: float(value)}, op=op)
            out[key] = float(reduced[key])
        return out


class ExportWriteCoordinator:
    """Single-writer coordination for ExportManager (manager unchanged)."""

    def __init__(self, sync: SyncFacade) -> None:
        self._sync = sync

    def should_write(self, context: DistributedContext) -> bool:
        policy: ExportWritePolicy = context.policy.export_write
        if policy.require_barrier_before_export:
            self._sync.barrier("export_before", policy=context.policy.barrier)
        return context.session.topology.global_rank == policy.writer_rank

    def run_writer(
        self,
        context: DistributedContext,
        write_fn: Callable[[], Any],
    ) -> Any:
        if not self.should_write(context):
            self._sync.barrier("export_after", policy=BarrierPolicy())
            return None
        try:
            result = write_fn()
        except Exception as exc:  # noqa: BLE001
            raise DistributedError(f"Export writer failed: {exc}") from exc
        self._sync.barrier("export_after", policy=BarrierPolicy())
        return result
