"""Packing runtime context — bindable bag for PackingStrategy adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from aiodoo_training.domain.config import PackingSpec
from aiodoo_training.domain.examples import TrainingExample
from aiodoo_training.domain.packing_policies import MemoryPackingPolicy, PackingPolicy
from aiodoo_training.domain.packing_session import PackingSession, PackingStatistics
from aiodoo_training.packing.token_rows import TokenRow


@dataclass(frozen=True, slots=True)
class PackingContext:
    """Resolved collaborators for packing (never framework types)."""

    examples: tuple[TrainingExample, ...]
    packing_session: PackingSession
    packing_spec: PackingSpec
    packing_policy: PackingPolicy
    token_rows: Mapping[str, TokenRow] = field(default_factory=dict)
    memory_policy: MemoryPackingPolicy = field(default_factory=MemoryPackingPolicy)
    packing_statistics: PackingStatistics | None = None
    overflow_deferred: int = 0
    overflow_truncated: int = 0
    seed: int = 42
    bind_extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_rows", MappingProxyType(dict(self.token_rows)))

    def with_session(self, session: PackingSession) -> PackingContext:
        return replace(self, packing_session=session)

    def with_statistics(self, statistics: PackingStatistics) -> PackingContext:
        return replace(self, packing_statistics=statistics)

    def with_overflow(self, *, deferred: int = 0, truncated: int = 0) -> PackingContext:
        return replace(
            self,
            overflow_deferred=self.overflow_deferred + deferred,
            overflow_truncated=self.overflow_truncated + truncated,
        )
