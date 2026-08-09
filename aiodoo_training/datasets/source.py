"""JSONL DatasetSource implementation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from types import MappingProxyType

from aiodoo_training.datasets.fingerprinting import fingerprint_dataset_mix
from aiodoo_training.datasets.fp2_example_loader import (
    iter_fp2_training_examples,
    validate_fp2_dataset_ref,
)
from aiodoo_training.datasets.mixing import mix_examples, stable_example_id
from aiodoo_training.datasets.reader import ProtocolRecordReader
from aiodoo_training.datasets.validation import DatasetValidator
from aiodoo_training.domain.config import DatasetMixSpec
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.examples import TrainingExample
from aiodoo_training.domain.refs import (
    RECORD_FORMAT_FP2_TRAINING_EXAMPLE,
    DatasetRef,
)
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.exceptions import DomainError
from aiodoo_training.ports.dataset import DatasetSource, ExampleFormatter
from aiodoo_training.registries import formatter_registry


class JsonlDatasetSource(DatasetSource):
    """
    Load and format protocol JSONL datasets into TrainingExample streams.

    Supports explicit ``DatasetRef.record_format``:
    - ``protocol_v1`` (default): Protocol V1 → ExampleFormatter projection
    - ``fp2_training_example``: identity load of FP2 TrainingExample packs
      (no Protocol formatter / aiodoo_contract projection)
    """

    def __init__(
        self,
        *,
        reader: ProtocolRecordReader | None = None,
        validator: DatasetValidator | None = None,
        validate: bool = True,
    ) -> None:
        self._reader = reader or ProtocolRecordReader()
        self._validator = validator or DatasetValidator(self._reader)
        self._validate = validate

    def load(self, refs: Sequence[DatasetRef]) -> Iterator[TrainingExample]:
        for ref in refs:
            yield from self._load_ref(ref)

    def load_mix(self, mix: DatasetMixSpec) -> Iterator[TrainingExample]:
        groups: list[tuple[TrainingExample, ...]] = []
        weights: list[float] = []
        for ref in mix.datasets:
            examples = tuple(self._load_ref(ref))
            groups.append(examples)
            weights.append(ref.weight)
        mixed = mix_examples(
            groups,
            weights=weights,
            shuffle=mix.shuffle,
            seed=mix.seed,
        )
        yield from mixed

    def open_session(
        self,
        mix: DatasetMixSpec,
        *,
        session_id: str,
        worker_id: int = 0,
        world_size: int = 1,
        global_rank: int = 0,
        local_rank: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
    ) -> tuple[DatasetSession, tuple[TrainingExample, ...]]:
        """Materialize a mix and return an immutable DatasetSession snapshot."""
        if self._validate:
            for ref in mix.datasets:
                self._validate_ref(ref)
        fingerprint = fingerprint_dataset_mix(
            mix.datasets,
            shuffle=mix.shuffle,
            seed=mix.seed,
            reader=self._reader,
        )
        examples = tuple(self.load_mix(mix))
        session = DatasetSession(
            session_id=session_id,
            dataset_fingerprint=fingerprint,
            mix_fingerprint=fingerprint,
            examples_total=len(examples),
            shuffle_seed=mix.seed if mix.shuffle else None,
            worker_id=worker_id,
            world_size=world_size,
            global_rank=global_rank,
            local_rank=local_rank,
            shard_id=shard_id,
            num_shards=num_shards,
            metadata=MappingProxyType({"example_count": len(examples)}),
        )
        return session, examples

    def _validate_ref(self, ref: DatasetRef) -> None:
        if ref.record_format == RECORD_FORMAT_FP2_TRAINING_EXAMPLE:
            validate_fp2_dataset_ref(ref)
            return
        self._validator.validate_ref(ref)

    def _load_ref(self, ref: DatasetRef) -> Iterator[TrainingExample]:
        if self._validate:
            self._validate_ref(ref)
        if ref.record_format == RECORD_FORMAT_FP2_TRAINING_EXAMPLE:
            # Identity path — never touch Protocol V1 formatters.
            yield from iter_fp2_training_examples(
                ref, reader=self._reader, validate=self._validate
            )
            return
        formatter = self._resolve_formatter(ref.dataset_type)
        path = Path(ref.path)
        for index, record in enumerate(self._reader.iter_records(path)):
            example = formatter.format(record, ref.dataset_type.value)
            if not example.example_id:
                example = TrainingExample(
                    example_id=stable_example_id(ref.dataset_type.value, record, index),
                    dataset_type=example.dataset_type,
                    messages=example.messages,
                    metadata=example.metadata,
                )
            yield example

    def _resolve_formatter(self, dataset_type: DatasetType) -> ExampleFormatter:
        key = dataset_type.value
        if not formatter_registry.exists(key):
            raise DomainError(
                f"No ExampleFormatter registered for dataset type '{key}'. "
                f"Known: {', '.join(formatter_registry.list()) or '(none)'}."
            )
        formatter_cls = formatter_registry.get(key)
        return formatter_cls()
