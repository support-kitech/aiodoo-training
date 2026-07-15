"""EpochCoordinator — align epoch boundaries across ranks via barriers."""

from __future__ import annotations

from aiodoo_training.distributed.sync import SyncFacade
from aiodoo_training.domain.distributed_policies import BarrierPolicy
from aiodoo_training.domain.session import DatasetSession


class EpochCoordinator:
    """Barrier then DatasetSession.next_epoch() on each rank."""

    def __init__(self, sync: SyncFacade) -> None:
        self._sync = sync

    def next_epoch(
        self,
        session: DatasetSession,
        *,
        barrier_policy: BarrierPolicy | None = None,
    ) -> DatasetSession:
        self._sync.barrier("epoch", policy=barrier_policy)
        return session.next_epoch()
