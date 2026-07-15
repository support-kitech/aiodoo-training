"""Runtime dataset consumption session (frozen Phase 0 abstraction)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from aiodoo_training.domain.identifiers import ExperimentId, RunId


@dataclass(frozen=True, slots=True)
class DatasetSession:
    """
    Immutable runtime state for dataset consumption.

    Owns iterator position, epoch, shuffle, shard/worker assignment, resume
    metadata, fingerprints, and progress — without tokenization or training
    logic. Designed for single-process streaming today and multi-node /
    checkpoint restore later without domain redesign.

    Updates must use copy-on-write helpers (:meth:`advance`, :meth:`next_epoch`,
    :meth:`with_fingerprint`, :meth:`with_progress`).
    """

    session_id: str
    experiment_id: ExperimentId | None = None
    run_id: RunId | None = None
    dataset_fingerprint: str | None = None
    mix_fingerprint: str | None = None
    epoch: int = 0
    example_index: int = 0
    examples_seen: int = 0
    examples_total: int | None = None
    shuffle_seed: int | None = None
    # Process / distributed placement
    worker_id: int = 0
    world_size: int = 1
    global_rank: int = 0
    local_rank: int = 0
    shard_id: int = 0
    num_shards: int = 1
    resume_token: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("DatasetSession.session_id must be a non-empty string.")
        if self.epoch < 0:
            raise ValueError("DatasetSession.epoch must be >= 0.")
        if self.example_index < 0:
            raise ValueError("DatasetSession.example_index must be >= 0.")
        if self.examples_seen < 0:
            raise ValueError("DatasetSession.examples_seen must be >= 0.")
        if self.examples_total is not None and self.examples_total < 0:
            raise ValueError("DatasetSession.examples_total must be >= 0 when set.")
        if self.worker_id < 0:
            raise ValueError("DatasetSession.worker_id must be >= 0.")
        if self.world_size < 1:
            raise ValueError("DatasetSession.world_size must be >= 1.")
        if self.global_rank < 0:
            raise ValueError("DatasetSession.global_rank must be >= 0.")
        if self.local_rank < 0:
            raise ValueError("DatasetSession.local_rank must be >= 0.")
        if self.shard_id < 0:
            raise ValueError("DatasetSession.shard_id must be >= 0.")
        if self.num_shards < 1:
            raise ValueError("DatasetSession.num_shards must be >= 1.")
        if self.shard_id >= self.num_shards:
            raise ValueError("DatasetSession.shard_id must be < num_shards.")

    def advance(self, *, steps: int = 1) -> DatasetSession:
        """Return a new session advanced by ``steps`` examples within the epoch."""
        if steps < 0:
            raise ValueError("steps must be >= 0.")
        return replace(
            self,
            example_index=self.example_index + steps,
            examples_seen=self.examples_seen + steps,
        )

    def next_epoch(self) -> DatasetSession:
        """Return a new session at the start of the next epoch."""
        return replace(self, epoch=self.epoch + 1, example_index=0)

    def with_fingerprint(self, fingerprint: str) -> DatasetSession:
        """Return a new session with an updated dataset fingerprint."""
        return replace(self, dataset_fingerprint=fingerprint)

    def with_mix_fingerprint(self, fingerprint: str) -> DatasetSession:
        """Return a new session with an updated mix fingerprint."""
        return replace(self, mix_fingerprint=fingerprint)

    def with_progress(self, **updates: Any) -> DatasetSession:
        """Return a new session with selected fields replaced (re-validated)."""
        return replace(self, **updates)
