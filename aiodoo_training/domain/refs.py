"""Reference types pointing at external artifacts without loading them."""

from dataclasses import dataclass
from pathlib import Path

from aiodoo_training.domain.enums import (
    AdapterType,
    DatasetType,
    ModelFamily,
    Precision,
)


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """Reference to a protocol dataset produced by aiodoo-datasets."""

    path: Path
    dataset_type: DatasetType
    protocol_version: str
    checksum: str | None = None
    weight: float = 1.0
    name: str | None = None


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
