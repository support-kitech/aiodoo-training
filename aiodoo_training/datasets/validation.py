"""Dataset validation against protocol expectations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aiodoo_training.datasets.reader import ProtocolRecordReader
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.exceptions import DomainError

REQUIRED_FIELDS: dict[DatasetType, frozenset[str]] = {
    DatasetType.PLANNER: frozenset({"instruction", "output", "metadata"}),
    DatasetType.CODING: frozenset({"instruction", "output", "metadata"}),
    DatasetType.REPAIR: frozenset({"instruction", "output", "metadata"}),
    DatasetType.EXECUTION: frozenset({"instruction", "output", "metadata"}),
    DatasetType.CONVERSATION: frozenset({"instruction", "output", "metadata"}),
    DatasetType.CONTEXT: frozenset({"id", "query", "metadata"}),
    DatasetType.APPROVAL: frozenset({"review_id", "decision", "metadata"}),
    DatasetType.EVALUATION: frozenset({"evaluation_id", "metadata"}),
    DatasetType.MIXED: frozenset(),
}


class DatasetValidator:
    """Fail-fast validation for DatasetRef targets before training consume."""

    def __init__(self, reader: ProtocolRecordReader | None = None) -> None:
        self._reader = reader or ProtocolRecordReader()

    def validate_ref(self, ref: DatasetRef, *, sample_limit: int = 32) -> None:
        """
        Validate that ``ref.path`` exists, is readable JSONL, and matches type rules.

        Raises:
            DomainError: on any contract violation.
        """
        path = Path(ref.path)
        if not path.exists():
            raise DomainError(f"Dataset path does not exist: {path}")
        if ref.protocol_version.strip() == "":
            raise DomainError("DatasetRef.protocol_version must be non-empty.")

        manifest_path = path.with_name(path.stem + "_manifest.json")
        if not manifest_path.exists():
            # Alternate convention: sibling manifest.json next to dataset
            alt = path.with_name(path.stem.replace("_dataset", "") + "_manifest.json")
            if alt.exists():
                manifest_path = alt

        if manifest_path.exists():
            self._validate_manifest(manifest_path, ref)

        required = REQUIRED_FIELDS.get(ref.dataset_type, frozenset())
        seen = 0
        for record in self._reader.iter_records(path):
            seen += 1
            missing = required - record.keys()
            if missing:
                raise DomainError(
                    f"Dataset {path} record missing required fields "
                    f"for {ref.dataset_type.value}: {sorted(missing)}"
                )
            if seen >= sample_limit:
                break
        if seen == 0:
            raise DomainError(f"Dataset is empty: {path}")

    def _validate_manifest(self, manifest_path: Path, ref: DatasetRef) -> None:
        try:
            data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DomainError(f"Invalid dataset manifest {manifest_path}: {exc}") from exc
        protocol = str(data.get("protocol_version", "")).strip()
        if protocol and protocol != ref.protocol_version:
            raise DomainError(
                f"Protocol version mismatch for {ref.path}: "
                f"ref={ref.protocol_version!r} manifest={protocol!r}"
            )
