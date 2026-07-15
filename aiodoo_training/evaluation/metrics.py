"""Evaluation-scoped metrics collection, aggregation, and history."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from aiodoo_training.domain.training import MetricSnapshot


@dataclass(frozen=True, slots=True)
class MetricHistory:
    """Append-only evaluation metric history with optional JSONL persistence."""

    snapshots: tuple[MetricSnapshot, ...] = ()
    history_path: Path | None = None

    def append(self, snapshot: MetricSnapshot) -> MetricHistory:
        updated = replace(self, snapshots=self.snapshots + (snapshot,))
        if self.history_path is not None:
            _append_jsonl(self.history_path, snapshot)
        return updated

    def extend(self, snapshots: Sequence[MetricSnapshot]) -> MetricHistory:
        result = self
        for snapshot in snapshots:
            result = result.append(snapshot)
        return result

    def to_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for snapshot in self.snapshots:
                handle.write(json.dumps(_snapshot_to_dict(snapshot), sort_keys=True))
                handle.write("\n")

    def flush(self, path: Path | None = None) -> None:
        target = path if path is not None else self.history_path
        if target is None:
            return
        self.to_jsonl(target)


class MetricAggregator:
    """Window / full-pass metric aggregates for evaluation."""

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = {}

    def observe(self, snapshot: MetricSnapshot) -> None:
        self._values.setdefault(snapshot.name, []).append(snapshot.value)

    def mean(self, name: str) -> float | None:
        values = self._values.get(name)
        if not values:
            return None
        return sum(values) / len(values)

    def reset(self, name: str | None = None) -> None:
        if name is None:
            self._values.clear()
        else:
            self._values.pop(name, None)

    def aggregated_snapshots(
        self, *, step: int, timestamp: datetime | None = None
    ) -> tuple[MetricSnapshot, ...]:
        ts = timestamp or datetime.now(UTC)
        out: list[MetricSnapshot] = []
        for name, values in sorted(self._values.items()):
            if not values:
                continue
            out.append(
                MetricSnapshot(
                    name=f"{name}_mean",
                    value=sum(values) / len(values),
                    step=step,
                    timestamp=ts,
                )
            )
        return tuple(out)


class MetricCollector:
    """Receives evaluation observations and emits frozen :class:`MetricSnapshot`."""

    def __init__(self, history: MetricHistory | None = None) -> None:
        self._history = history or MetricHistory()
        self._aggregator = MetricAggregator()

    @property
    def history(self) -> MetricHistory:
        return self._history

    @property
    def aggregator(self) -> MetricAggregator:
        return self._aggregator

    def observe(self, snapshot: MetricSnapshot) -> None:
        self._history = self._history.append(snapshot)
        self._aggregator.observe(snapshot)

    def observe_many(self, snapshots: Sequence[MetricSnapshot]) -> None:
        for snapshot in snapshots:
            self.observe(snapshot)

    def snapshot(
        self,
        *,
        name: str,
        value: float,
        step: int,
        timestamp: datetime | None = None,
    ) -> MetricSnapshot:
        return MetricSnapshot(
            name=name,
            value=value,
            step=step,
            timestamp=timestamp or datetime.now(UTC),
        )


def _snapshot_to_dict(snapshot: MetricSnapshot) -> dict[str, object]:
    return {
        "name": snapshot.name,
        "value": snapshot.value,
        "step": snapshot.step,
        "timestamp": snapshot.timestamp.isoformat() if snapshot.timestamp else None,
        "tags": list(snapshot.tags),
    }


def _append_jsonl(path: Path, snapshot: MetricSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_snapshot_to_dict(snapshot), sort_keys=True))
        handle.write("\n")
