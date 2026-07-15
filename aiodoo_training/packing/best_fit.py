"""Best-fit packing — O(n log n) via sorted residuals + bisect."""

from __future__ import annotations

import bisect
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
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


@dataclass
class _Bin:
    ids: list[int] = field(default_factory=list)
    mask: list[int] = field(default_factory=list)
    labs: list[int] = field(default_factory=list)
    spans: list[PackedSpan] = field(default_factory=list)
    ex: list[str] = field(default_factory=list)


class BestFitPacking(PackingStrategy):
    """
    Length-descending best-fit into open sequences.

    Open bins keyed by residual capacity; bisect finds the tightest fit in
    O(log B) per placement after an O(n log n) sort — not O(n²).
    """

    BACKEND_KEY = "best_fit"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context

    def bind(self, context: Any) -> BestFitPacking:
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

        # Sort by length descending, tie-break example_id ascending (stable/deterministic).
        ordered = sorted(
            examples,
            key=lambda e: (-rows[e.example_id].length, e.example_id),
        )

        bins: list[_Bin] = []
        # sorted residual capacities aligned with residual_index -> bin_idx
        residuals: list[int] = []
        residual_to_bin: list[int] = []

        deferred = 0
        truncated = 0

        def remove_residual(pos: int) -> None:
            residuals.pop(pos)
            residual_to_bin.pop(pos)

        def insert_residual(residual: int, bin_idx: int) -> None:
            pos = bisect.bisect_left(residuals, residual)
            residuals.insert(pos, residual)
            residual_to_bin.insert(pos, bin_idx)

        for example in ordered:
            row = rows[example.example_id]
            need = row.length
            if need > max_len:
                if overflow == PackingOverflow.REJECT:
                    raise PackingError(
                        f"Example {example.example_id!r} length {need} exceeds max {max_len}."
                    )
                if overflow == PackingOverflow.TRUNCATE:
                    truncated += 1
                    ids = row.input_ids[:max_len]
                    mask = row.attention_mask[:max_len]
                    labs = row.labels[:max_len]
                    need = max_len
                else:
                    deferred += 1
                    ids = row.input_ids[:max_len]
                    mask = row.attention_mask[:max_len]
                    labs = row.labels[:max_len]
                    need = len(ids)
                bins.append(
                    _Bin(
                        ids=list(ids),
                        mask=list(mask),
                        labs=list(labs),
                        spans=[PackedSpan(example.example_id, 0, need)],
                        ex=[example.example_id],
                    )
                )
                continue
            else:
                ids = row.input_ids
                mask = row.attention_mask
                labs = row.labels

            placed = False
            lo = bisect.bisect_left(residuals, need)
            candidates = list(range(lo, len(residuals)))
            best_pos: int | None = None
            best_extra = 0
            for pos in candidates:
                bin_idx = residual_to_bin[pos]
                bin_state = bins[bin_idx]
                sep_cost = 1 if (sep is not None and bin_state.ids) else 0
                if max_ex is not None and len(bin_state.ex) >= max_ex:
                    continue
                if residuals[pos] >= need + sep_cost:
                    best_pos = pos
                    best_extra = sep_cost
                    break
            if best_pos is not None:
                bin_idx = residual_to_bin[best_pos]
                remove_residual(best_pos)
                bin_state = bins[bin_idx]
                start = len(bin_state.ids)
                if best_extra and sep is not None:
                    bin_state.ids.append(int(sep))
                    bin_state.mask.append(1)
                    bin_state.labs.append(IGNORE_INDEX)
                    start = len(bin_state.ids)
                bin_state.ids.extend(ids)
                bin_state.mask.extend(mask)
                bin_state.labs.extend(labs)
                bin_state.spans.append(PackedSpan(example.example_id, start, start + need))
                bin_state.ex.append(example.example_id)
                new_residual = max_len - len(bin_state.ids)
                if new_residual > 0:
                    insert_residual(new_residual, bin_idx)
                placed = True

            if not placed:
                bins.append(
                    _Bin(
                        ids=list(ids),
                        mask=list(mask),
                        labs=list(labs),
                        spans=[PackedSpan(example.example_id, 0, need)],
                        ex=[example.example_id],
                    )
                )
                residual = max_len - need
                if residual > 0:
                    insert_residual(residual, len(bins) - 1)

        if ctx is not None and hasattr(ctx, "with_overflow"):
            self._context = ctx.with_overflow(deferred=deferred, truncated=truncated)

        sequences = [
            (
                tuple(b.ids),
                tuple(b.mask),
                tuple(b.labs),
                tuple(b.spans),
                tuple(b.ex),
            )
            for b in bins
            if b.ids
        ]
        # Stable order by first example_id for determinism of batch row order
        sequences.sort(key=lambda s: s[4][0] if s[4] else "")
        if not sequences:
            return
            yield  # pragma: no cover
        yield emit_sequences(sequences, width=width)


def register_best_fit_packing(*, overwrite: bool = False) -> None:
    from aiodoo_training.registries import packing_registry

    if not packing_registry.exists("best_fit") or overwrite:
        packing_registry.register("best_fit", BestFitPacking, overwrite=overwrite)
