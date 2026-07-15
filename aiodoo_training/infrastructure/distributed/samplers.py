"""DistributedSampler implementations (deterministic, framework-free)."""

from __future__ import annotations

from collections.abc import Sequence

from aiodoo_training.distributed.seed import derive_rank_seed
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.ports.distributed import DistributedSampler


class ShardDistributedSampler(DistributedSampler):
    """Contiguous shard of indices for this rank, optionally shuffled by seed."""

    def sample_indices(
        self,
        total: int,
        session: DatasetSession,
        seed: int,
    ) -> Sequence[int]:
        if total < 0:
            raise ValueError("total must be >= 0.")
        world = max(1, session.world_size)
        rank = session.global_rank
        # Contiguous shards; remainder to last ranks.
        base = total // world
        rem = total % world
        start = rank * base + min(rank, rem)
        count = base + (1 if rank < rem else 0)
        indices = list(range(start, start + count))
        # Deterministic shuffle within shard
        rng_seed = derive_rank_seed(seed, rank, epoch=session.epoch)
        return _deterministic_shuffle(indices, rng_seed)


def _deterministic_shuffle(items: list[int], seed: int) -> list[int]:
    # Fisher–Yates with portable LCG
    arr = list(items)
    state = seed % (2**31 - 1) or 1
    for i in range(len(arr) - 1, 0, -1):
        state = (1103515245 * state + 12345) % (2**31)
        j = state % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def register_default_distributed_samplers(*, overwrite: bool = False) -> None:
    from aiodoo_training.registries import distributed_sampler_registry

    distributed_sampler_registry.register("shard", ShardDistributedSampler, overwrite=overwrite)
