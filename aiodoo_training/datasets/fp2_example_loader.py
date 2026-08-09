"""FP2 TrainingExample JSONL loader (AT-2).

Loads packs that are already TrainingExample-shaped. Does **not** run
Protocol V1 ExampleFormatters or aiodoo_contract projection.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from aiodoo_training.datasets.reader import ProtocolRecordReader
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.examples import TrainingExample, freeze_messages
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.exceptions import DomainError
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

REQUIRED_FP2_KEYS: frozenset[str] = frozenset(
    {"example_id", "dataset_type", "messages"}
)


def validate_fp2_training_example_record(
    record: Mapping[str, Any],
    *,
    path: Path,
    line_no: int,
) -> None:
    """Raise DomainError if ``record`` is not a well-formed FP2 TrainingExample row."""
    missing = sorted(REQUIRED_FP2_KEYS - set(record.keys()))
    if missing:
        raise DomainError(
            f"FP2 TrainingExample missing keys {missing} at {path}:{line_no}"
        )
    example_id = record.get("example_id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise DomainError(f"FP2 TrainingExample example_id invalid at {path}:{line_no}")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise DomainError(
            f"FP2 TrainingExample messages must be a list of >=2 turns at {path}:{line_no}"
        )
    for idx, msg in enumerate(messages):
        if not isinstance(msg, Mapping):
            raise DomainError(
                f"FP2 TrainingExample messages[{idx}] must be an object at {path}:{line_no}"
            )
        if not str(msg.get("role") or "").strip():
            raise DomainError(
                f"FP2 TrainingExample messages[{idx}].role required at {path}:{line_no}"
            )
        if "content" not in msg:
            raise DomainError(
                f"FP2 TrainingExample messages[{idx}].content required at {path}:{line_no}"
            )
    meta = record.get("metadata")
    if meta is not None and not isinstance(meta, Mapping):
        raise DomainError(f"FP2 TrainingExample metadata must be an object at {path}:{line_no}")


def record_to_training_example(
    record: Mapping[str, Any],
    *,
    fallback_dataset_type: DatasetType,
) -> TrainingExample:
    """Convert a validated FP2 pack row into a domain TrainingExample."""
    type_raw = str(record.get("dataset_type") or fallback_dataset_type.value)
    try:
        dataset_type = DatasetType(type_raw)
    except ValueError:
        dataset_type = fallback_dataset_type
    messages = freeze_messages(tuple(record["messages"]))  # type: ignore[arg-type]
    meta_raw = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    meta = dict(meta_raw)
    meta.setdefault("fp2_native", True)
    meta.setdefault("record_format", "fp2_training_example")
    # Preserve contract identity when present; never invent Protocol projection.
    if "training_contract_version" not in meta:
        meta["training_contract_version"] = SYSTEM_TRAINING_CONTRACT_VERSION
    return TrainingExample(
        example_id=str(record["example_id"]).strip(),
        dataset_type=dataset_type,
        messages=messages,
        metadata=MappingProxyType(meta),
    )


def iter_fp2_training_examples(
    ref: DatasetRef,
    *,
    reader: ProtocolRecordReader | None = None,
    validate: bool = True,
) -> Iterator[TrainingExample]:
    """Stream TrainingExamples from an FP2 pack JSONL referenced by ``ref``."""
    path = Path(ref.path)
    stream = reader or ProtocolRecordReader()
    for line_no, record in enumerate(stream.iter_records(path), start=1):
        if validate:
            validate_fp2_training_example_record(record, path=path, line_no=line_no)
        yield record_to_training_example(record, fallback_dataset_type=ref.dataset_type)


def validate_fp2_dataset_ref(ref: DatasetRef, *, sample_limit: int = 8) -> int:
    """Validate the first ``sample_limit`` rows of an FP2 pack; return total count."""
    path = Path(ref.path)
    if not path.is_file():
        raise DomainError(f"FP2 dataset file not found: {path}")
    count = 0
    for line_no, record in enumerate(ProtocolRecordReader().iter_records(path), start=1):
        count += 1
        if line_no <= sample_limit:
            validate_fp2_training_example_record(record, path=path, line_no=line_no)
    if count == 0:
        raise DomainError(f"FP2 dataset is empty: {path}")
    return count
