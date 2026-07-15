"""Shared packing helpers — deterministic token rows and TokenBatch emission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from aiodoo_training.domain.examples import IGNORE_INDEX, TokenBatch, TrainingExample
from aiodoo_training.domain.packing_session import PackedSpan


@dataclass(frozen=True, slots=True)
class TokenRow:
    """Per-example token materialisation for packing (framework-free)."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.input_ids)


def build_stub_token_row(example: TrainingExample, *, max_length: int) -> TokenRow:
    """
    Deterministic CPU stub tokens from example identity + message content.

    No framework imports. Length is bounded by ``max_length``.
    """
    parts: list[str] = [example.example_id]
    for message in example.messages:
        parts.append(str(message.get("content", "")))
    material = "|".join(parts)
    # Map bytes deterministically into a small positive id space.
    ids = [((b % 97) + 1) for b in material.encode("utf-8")]
    if not ids:
        ids = [1]
    if len(ids) > max_length:
        ids = ids[:max_length]
    input_ids = tuple(ids)
    attention_mask = tuple(1 for _ in input_ids)
    labels = tuple(input_ids)
    return TokenRow(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def resolve_token_rows(
    examples: Sequence[TrainingExample],
    *,
    max_length: int,
    provided: Mapping[str, TokenRow] | None = None,
) -> dict[str, TokenRow]:
    """Use provided rows when present; otherwise build stub rows."""
    out: dict[str, TokenRow] = {}
    for example in examples:
        if provided is not None and example.example_id in provided:
            out[example.example_id] = provided[example.example_id]
        else:
            out[example.example_id] = build_stub_token_row(example, max_length=max_length)
    return out


def pad_row(row: TokenRow, *, width: int, pad_id: int = 0) -> TokenRow:
    if len(row.input_ids) > width:
        return TokenRow(
            input_ids=row.input_ids[:width],
            attention_mask=row.attention_mask[:width],
            labels=row.labels[:width],
        )
    pad_n = width - len(row.input_ids)
    if pad_n <= 0:
        return row
    return TokenRow(
        input_ids=row.input_ids + (pad_id,) * pad_n,
        attention_mask=row.attention_mask + (0,) * pad_n,
        labels=row.labels + (IGNORE_INDEX,) * pad_n,
    )


PackedSequence = tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[PackedSpan, ...],
    tuple[str, ...],
]


def emit_sequences(
    sequences: Sequence[PackedSequence],
    *,
    width: int,
) -> TokenBatch:
    """
    Build one rectangular TokenBatch from packed sequences.

    Each sequence is (input_ids, attention_mask, labels, spans, example_ids).
    """
    if not sequences:
        return TokenBatch(
            example_ids=(),
            input_ids=(),
            attention_mask=(),
            labels=(),
            metadata=MappingProxyType({"packed_spans": ()}),
        )

    example_ids: list[str] = []
    input_ids: list[tuple[int, ...]] = []
    attention_mask: list[tuple[int, ...]] = []
    labels: list[tuple[int, ...]] = []
    all_spans: list[tuple[PackedSpan, ...]] = []

    for ids, mask, labs, spans, ex_ids in sequences:
        row = pad_row(
            TokenRow(input_ids=ids, attention_mask=mask, labels=labs),
            width=width,
        )
        # One batch row per packed sequence; join example ids for identity.
        example_ids.append("+".join(ex_ids) if ex_ids else "empty")
        input_ids.append(row.input_ids)
        attention_mask.append(row.attention_mask)
        labels.append(row.labels)
        all_spans.append(spans)

    return TokenBatch(
        example_ids=tuple(example_ids),
        input_ids=tuple(input_ids),
        attention_mask=tuple(attention_mask),
        labels=tuple(labels),
        metadata=MappingProxyType(
            {
                "packed_spans": tuple(
                    tuple(
                        {
                            "example_id": s.example_id,
                            "start": s.start,
                            "end": s.end,
                        }
                        for s in spans
                    )
                    for spans in all_spans
                )
            }
        ),
    )


def round_up(length: int, multiple: int | None) -> int:
    if multiple is None or multiple <= 1:
        return length
    rem = length % multiple
    if rem == 0:
        return length
    return length + (multiple - rem)
