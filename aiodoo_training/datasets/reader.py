"""Streaming JSONL protocol record reader."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from aiodoo_training.exceptions import DomainError


class ProtocolRecordReader:
    """
    Stream protocol JSONL files as plain dictionaries.

    Does not interpret generator-specific schemas — formatters own that.
    """

    def __init__(self, *, encoding: str = "utf-8") -> None:
        self._encoding = encoding

    def iter_records(self, path: Path) -> Iterator[dict[str, Any]]:
        """Yield each JSON object from ``path`` in file order."""
        if not path.exists():
            raise DomainError(f"Dataset file not found: {path}")
        if not path.is_file():
            raise DomainError(f"Dataset path is not a file: {path}")

        with path.open("r", encoding=self._encoding) as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DomainError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if not isinstance(record, dict):
                    raise DomainError(f"JSONL record must be an object at {path}:{line_no}")
                yield record

    def count_records(self, path: Path) -> int:
        """Count non-empty JSONL records without materializing them."""
        return sum(1 for _ in self.iter_records(path))

    def content_sha256(self, path: Path) -> str:
        """Hash raw file bytes for deterministic fingerprints."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
