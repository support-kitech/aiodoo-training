"""Operational logging — separate from ExperimentTracker."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True, slots=True)
class LogRecord:
    level: str
    message: str
    fields: tuple[tuple[str, str], ...] = ()
    timestamp: str = ""


class LogSink:
    def emit(self, record: LogRecord) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class ConsoleLogSink(LogSink):
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stderr

    def emit(self, record: LogRecord) -> None:
        extras = " ".join(f"{k}={v}" for k, v in record.fields)
        line = f"[{record.level}] {record.message}"
        if extras:
            line = f"{line} {extras}"
        print(line, file=self._stream)


class JsonlLogSink(LogSink):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: LogRecord) -> None:
        payload = {
            "level": record.level,
            "message": record.message,
            "fields": dict(record.fields),
            "timestamp": record.timestamp,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


@dataclass
class Logger:
    """Simple structured logger facade."""

    sinks: list[LogSink] = field(default_factory=list)
    bound: dict[str, str] = field(default_factory=dict)
    level: str = "INFO"

    _ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

    def bind(self, **fields: Any) -> Logger:
        merged = {**self.bound, **{str(k): str(v) for k, v in fields.items()}}
        return Logger(sinks=list(self.sinks), bound=merged, level=self.level)

    def _emit(self, level: str, message: str) -> None:
        if self._ORDER.get(level, 20) < self._ORDER.get(self.level.upper(), 20):
            return
        record = LogRecord(
            level=level,
            message=message,
            fields=tuple(sorted(self.bound.items())),
            timestamp=datetime.now(UTC).isoformat(),
        )
        for sink in self.sinks:
            try:
                sink.emit(record)
            except Exception:  # noqa: BLE001 — logging never fatal
                continue

    def debug(self, message: str) -> None:
        self._emit("DEBUG", message)

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warning(self, message: str) -> None:
        self._emit("WARNING", message)

    def error(self, message: str) -> None:
        self._emit("ERROR", message)


def build_default_logger(*, jsonl_path: Path | None = None, level: str = "INFO") -> Logger:
    sinks: list[LogSink] = [ConsoleLogSink()]
    if jsonl_path is not None:
        sinks.append(JsonlLogSink(jsonl_path))
    return Logger(sinks=sinks, level=level)
