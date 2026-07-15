"""Greedy concatenation packing — O(n)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from aiodoo_training.domain.config import PackingSpec
from aiodoo_training.domain.enums import PackingOverflow
from aiodoo_training.domain.examples import IGNORE_INDEX, TokenBatch, TrainingExample
from aiodoo_training.domain.packing_session import PackedSpan
from aiodoo_training.exceptions import PackingError
from aiodoo_training.packing.token_rows import (
    emit_sequences,
    resolve_token_rows,
    round_up,
)
from aiodoo_training.ports.packing import PackingStrategy


class ConcatenationPacking(PackingStrategy):
    """Single-pass greedy concatenation into sequences of max_sequence_length."""

    BACKEND_KEY = "concat"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context

    def bind(self, context: Any) -> ConcatenationPacking:
        self._context = context
        return self

    def pack(
        self,
        examples: Sequence[TrainingExample],
        spec: PackingSpec,
    ) -> Iterator[TokenBatch]:
        ctx = self._context
        policy = getattr(ctx, "packing_policy", None)
        max_len = int(policy.max_sequence_length if policy else spec.max_sequence_length)
        max_ex = policy.max_examples_per_sequence if policy else None
        sep = policy.separator_token_id if policy else None
        overflow = policy.overflow if policy else PackingOverflow.DEFER
        pad_mult = policy.pad_to_multiple_of if policy else None
        width = round_up(max_len, pad_mult)

        provided = dict(getattr(ctx, "token_rows", {}) or {})
        rows = resolve_token_rows(examples, max_length=max_len, provided=provided)

        sequences: list[
            tuple[
                tuple[int, ...],
                tuple[int, ...],
                tuple[int, ...],
                tuple[PackedSpan, ...],
                tuple[str, ...],
            ]
        ] = []
        cur_ids: list[int] = []
        cur_mask: list[int] = []
        cur_labs: list[int] = []
        cur_spans: list[PackedSpan] = []
        cur_ex: list[str] = []
        deferred = 0
        truncated = 0

        def flush() -> None:
            nonlocal cur_ids, cur_mask, cur_labs, cur_spans, cur_ex
            if not cur_ids:
                return
            sequences.append(
                (
                    tuple(cur_ids),
                    tuple(cur_mask),
                    tuple(cur_labs),
                    tuple(cur_spans),
                    tuple(cur_ex),
                )
            )
            cur_ids, cur_mask, cur_labs, cur_spans, cur_ex = [], [], [], [], []

        for example in examples:
            row = rows[example.example_id]
            need = len(row.input_ids)
            sep_cost = 1 if (sep is not None and cur_ids) else 0
            if need > max_len:
                if overflow == PackingOverflow.REJECT:
                    raise PackingError(
                        f"Example {example.example_id!r} length {need} exceeds max {max_len}."
                    )
                if overflow == PackingOverflow.TRUNCATE:
                    truncated += 1
                    row_ids = row.input_ids[:max_len]
                    row_mask = row.attention_mask[:max_len]
                    row_labs = row.labels[:max_len]
                    need = max_len
                else:
                    deferred += 1
                    flush()
                    # Defer: place truncated-to-max alone
                    row_ids = row.input_ids[:max_len]
                    row_mask = row.attention_mask[:max_len]
                    row_labs = row.labels[:max_len]
                    need = len(row_ids)
                    cur_ids.extend(row_ids)
                    cur_mask.extend(row_mask)
                    cur_labs.extend(row_labs)
                    cur_spans.append(PackedSpan(example.example_id, 0, need))
                    cur_ex.append(example.example_id)
                    flush()
                    continue
            else:
                row_ids = row.input_ids
                row_mask = row.attention_mask
                row_labs = row.labels

            if cur_ids and (
                len(cur_ids) + sep_cost + need > max_len
                or (max_ex is not None and len(cur_ex) >= max_ex)
            ):
                flush()
                sep_cost = 0

            start = len(cur_ids)
            if sep_cost and sep is not None:
                cur_ids.append(int(sep))
                cur_mask.append(1)
                cur_labs.append(IGNORE_INDEX)
                start = len(cur_ids)
            cur_ids.extend(row_ids)
            cur_mask.extend(row_mask)
            cur_labs.extend(row_labs)
            cur_spans.append(PackedSpan(example.example_id, start, start + need))
            cur_ex.append(example.example_id)

        flush()
        if ctx is not None and hasattr(ctx, "with_overflow"):
            self._context = ctx.with_overflow(deferred=deferred, truncated=truncated)

        if not sequences:
            return
            yield  # pragma: no cover — makes this a generator
        yield emit_sequences(sequences, width=width)


def register_concat_packing(*, overwrite: bool = False) -> None:
    from aiodoo_training.registries import packing_registry

    if not packing_registry.exists("concat") or overwrite:
        packing_registry.register("concat", ConcatenationPacking, overwrite=overwrite)
