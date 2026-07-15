"""Dataset fingerprinting."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from aiodoo_training.datasets.reader import ProtocolRecordReader
from aiodoo_training.domain.refs import DatasetRef


def fingerprint_dataset_file(path: Path, reader: ProtocolRecordReader | None = None) -> str:
    """SHA-256 of raw dataset file bytes."""
    active = reader or ProtocolRecordReader()
    return active.content_sha256(path)


def fingerprint_dataset_ref(ref: DatasetRef, reader: ProtocolRecordReader | None = None) -> str:
    """Fingerprint a single DatasetRef (type + protocol + content)."""
    content = fingerprint_dataset_file(Path(ref.path), reader)
    material = (
        f"type={ref.dataset_type.value}|protocol={ref.protocol_version}|"
        f"weight={ref.weight}|content={content}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def fingerprint_dataset_mix(
    refs: Sequence[DatasetRef],
    *,
    shuffle: bool,
    seed: int,
    reader: ProtocolRecordReader | None = None,
) -> str:
    """Deterministic fingerprint for a dataset mix specification."""
    parts = [
        f"shuffle={int(shuffle)}",
        f"seed={seed}",
    ]
    for ref in sorted(refs, key=lambda r: (r.dataset_type.value, str(r.path), r.weight)):
        parts.append(fingerprint_dataset_ref(ref, reader))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
