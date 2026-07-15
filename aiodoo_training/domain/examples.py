"""Tokenization configuration and expanded token batch domain types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from aiodoo_training.domain.enums import DatasetType

IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """
    Backend-agnostic training example after protocol formatting.

    ``messages`` use OpenAI-style role/content maps (``role``, ``content``).
    """

    example_id: str
    dataset_type: DatasetType
    messages: tuple[Mapping[str, str], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class TokenizationConfig:
    """Deterministic tokenization policy (feeds cache and fingerprint keys)."""

    max_length: int = 2048
    padding: str = "max_length"  # max_length | longest | do_not_pad
    truncation: bool = True
    mask_prompt: bool = True
    ignore_index: int = IGNORE_INDEX
    chat_template_name: str = "qwen"
    add_generation_prompt: bool = False

    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError("TokenizationConfig.max_length must be >= 1.")
        if self.padding not in {"max_length", "longest", "do_not_pad"}:
            raise ValueError(f"Unsupported padding policy: {self.padding}")


@dataclass(frozen=True, slots=True)
class TokenBatch:
    """
    Immutable tokenized batch ready for a future trainer.

    All sequences in a batch share the same length after padding/truncation.
    """

    example_ids: tuple[str, ...]
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    labels: tuple[tuple[int, ...], ...]
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        n = len(self.example_ids)
        if not (len(self.input_ids) == len(self.attention_mask) == len(self.labels) == n):
            raise ValueError("TokenBatch sequence fields must share the same length.")
        if n == 0:
            return
        width = len(self.input_ids[0])
        for field_name, rows in (
            ("input_ids", self.input_ids),
            ("attention_mask", self.attention_mask),
            ("labels", self.labels),
        ):
            for row in rows:
                if len(row) != width:
                    raise ValueError(f"TokenBatch.{field_name} rows must be rectangular.")


def freeze_messages(messages: Sequence[Mapping[str, str]]) -> tuple[Mapping[str, str], ...]:
    """Copy messages into immutable MappingProxyType tuples."""
    frozen: list[Mapping[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if not role:
            raise ValueError("TrainingExample messages require a non-empty role.")
        frozen.append(MappingProxyType({"role": role, "content": content}))
    return tuple(frozen)
