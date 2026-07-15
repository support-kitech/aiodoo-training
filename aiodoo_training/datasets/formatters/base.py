"""Shared formatter helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from aiodoo_training.datasets.mixing import stable_example_id
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.examples import TrainingExample, freeze_messages
from aiodoo_training.ports.dataset import ExampleFormatter


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def user_assistant(
    dataset_type: DatasetType,
    record: Mapping[str, Any],
    *,
    user_text: str,
    assistant_text: str,
    index: int = 0,
) -> TrainingExample:
    """Build a standard user/assistant TrainingExample."""
    messages = freeze_messages(
        (
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        )
    )
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    return TrainingExample(
        example_id=stable_example_id(dataset_type.value, record, index),
        dataset_type=dataset_type,
        messages=messages,
        metadata=MappingProxyType(dict(metadata)),
    )


class BaseFormatter(ExampleFormatter):
    """Common ExampleFormatter scaffolding."""

    dataset_type: DatasetType

    def supports(self, dataset_type: str) -> bool:
        return dataset_type == self.dataset_type.value

    def format(self, record: dict[str, object], dataset_type: str) -> TrainingExample:
        if not self.supports(dataset_type):
            raise ValueError(
                f"{type(self).__name__} does not support dataset type {dataset_type!r}"
            )
        return self._format(record)

    def _format(self, record: Mapping[str, Any]) -> TrainingExample:
        raise NotImplementedError
