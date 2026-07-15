"""ShardPlanner — populate DatasetSession placement fields only (no schema change)."""

from __future__ import annotations

from aiodoo_training.domain.distributed_session import DistributedTopology
from aiodoo_training.domain.session import DatasetSession


class ShardPlanner:
    """Maps topology onto frozen DatasetSession placement fields via COW."""

    def apply(self, session: DatasetSession, topology: DistributedTopology) -> DatasetSession:
        world = topology.world_size
        rank = topology.global_rank
        return session.with_progress(
            world_size=world,
            global_rank=rank,
            local_rank=topology.local_rank,
            worker_id=rank,
            shard_id=rank,
            num_shards=world,
        )
