"""Phase 6 metric history domain / store types."""

from __future__ import annotations

from dataclasses import dataclass, replace

from aiodoo_training.domain.training import MetricSnapshot


@dataclass(frozen=True, slots=True)
class MetricSeries:
    """Named contiguous metric values + tags."""

    name: str
    values: tuple[MetricSnapshot, ...] = ()
    tags: tuple[tuple[str, str], ...] = ()

    def append(self, snapshot: MetricSnapshot) -> MetricSeries:
        return replace(self, values=(*self.values, snapshot))


@dataclass(frozen=True, slots=True)
class MetricTimeline:
    """Ordered step → MetricSnapshot view across series."""

    snapshots: tuple[MetricSnapshot, ...] = ()

    def append(self, snapshot: MetricSnapshot) -> MetricTimeline:
        return replace(self, snapshots=(*self.snapshots, snapshot))

    def sorted(self) -> MetricTimeline:
        ordered = tuple(sorted(self.snapshots, key=lambda s: (s.step, s.name)))
        return replace(self, snapshots=ordered)
