"""Phase 4 metric catalog domain — declarative metric identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class MetricAggregation(StrEnum):
    """How per-example observations roll up into a scalar."""

    MEAN = "mean"
    SUM = "sum"
    LAST = "last"


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Stable metric identity for training and evaluation catalogs."""

    name: str
    aggregation: MetricAggregation = MetricAggregation.MEAN
    higher_is_better: bool = True
    unit: str | None = None
    tags: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("MetricDefinition.name must be non-empty.")
        object.__setattr__(
            self,
            "tags",
            MappingProxyType({str(k): str(v) for k, v in self.tags.items()}),
        )
