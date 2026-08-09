"""Reference types pointing at external artifacts without loading them."""

from dataclasses import dataclass
from pathlib import Path

from aiodoo_training.domain.enums import (
    AdapterType,
    DatasetType,
    ModelFamily,
    Precision,
)


# Explicit record wire formats — never guess from JSON shape.
RECORD_FORMAT_PROTOCOL_V1: str = "protocol_v1"
RECORD_FORMAT_FP2_TRAINING_EXAMPLE: str = "fp2_training_example"
KNOWN_RECORD_FORMATS: frozenset[str] = frozenset(
    {RECORD_FORMAT_PROTOCOL_V1, RECORD_FORMAT_FP2_TRAINING_EXAMPLE}
)


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """Reference to a protocol or FP2 TrainingExample dataset."""

    path: Path
    dataset_type: DatasetType
    protocol_version: str
    checksum: str | None = None
    weight: float = 1.0
    name: str | None = None
    # Explicit input format (AT-2). Default preserves Protocol V1 formatter path.
    record_format: str = RECORD_FORMAT_PROTOCOL_V1

    def __post_init__(self) -> None:
        fmt = (self.record_format or RECORD_FORMAT_PROTOCOL_V1).strip().lower()
        if fmt not in KNOWN_RECORD_FORMATS:
            raise ValueError(
                f"DatasetRef.record_format must be one of {sorted(KNOWN_RECORD_FORMATS)}; "
                f"got {self.record_format!r}"
            )
        object.__setattr__(self, "record_format", fmt)


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Reference to a base causal language model."""

    identifier: str
    family: ModelFamily
    revision: str | None = None
    local_path: Path | None = None
    precision: Precision = Precision.BF16


@dataclass(frozen=True, slots=True)
class AdapterRef:
    """Reference to a trained or to-be-trained adapter artifact."""

    name: str
    adapter_type: AdapterType
    path: Path | None = None
    base_model: ModelRef | None = None
