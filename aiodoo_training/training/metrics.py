"""Training metrics collection, aggregation, and history."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from aiodoo_training.domain.training import MetricSnapshot
from aiodoo_training.domain.training_events import TrainingEvent, TrainingEventKind
from aiodoo_training.training.context import CallbackContext


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Append-only immutable metric history with optional JSONL persistence."""

    snapshots: tuple[MetricSnapshot, ...] = ()
    history_path: Path | None = None

    def append(self, snapshot: MetricSnapshot) -> TrainingHistory:
        updated = replace(self, snapshots=self.snapshots + (snapshot,))
        if self.history_path is not None:
            _append_jsonl(self.history_path, snapshot)
        return updated

    def extend(self, snapshots: Sequence[MetricSnapshot]) -> TrainingHistory:
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
        """Write the full history to ``path`` (or ``history_path`` if set)."""
        target = path if path is not None else self.history_path
        if target is None:
            return
        self.to_jsonl(target)


class MetricAggregator:
    """Window / epoch metric aggregates (mean loss, etc.)."""

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
        for name, values in self._values.items():
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
    """Receives training events and builds :class:`MetricSnapshot` records."""

    def __init__(self, history: TrainingHistory | None = None) -> None:
        self._history = history or TrainingHistory()
        self._aggregator = MetricAggregator()

    @property
    def history(self) -> TrainingHistory:
        return self._history

    @property
    def aggregator(self) -> MetricAggregator:
        return self._aggregator

    def on_event(self, event: TrainingEvent, context: CallbackContext) -> None:
        if event.kind == TrainingEventKind.LOSS_COMPUTED and event.loss is not None:
            snapshot = MetricSnapshot(
                name="loss",
                value=event.loss,
                step=event.global_step,
                timestamp=event.timestamp,
            )
            self._history = self._history.append(snapshot)
            self._aggregator.observe(snapshot)
            return

        if event.kind == TrainingEventKind.METRICS_AGGREGATED and event.metrics:
            self._history = self._history.extend(event.metrics)
            for snapshot in event.metrics:
                self._aggregator.observe(snapshot)

    def loss_snapshot(
        self, *, loss: float, step: int, timestamp: datetime | None = None
    ) -> MetricSnapshot:
        return MetricSnapshot(
            name="loss", value=loss, step=step, timestamp=timestamp or datetime.now(UTC)
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
