"""No-op / passthrough packing strategy (Phase 1 + Phase 5)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from aiodoo_training.domain.config import PackingSpec
from aiodoo_training.domain.examples import TokenBatch, TrainingExample
from aiodoo_training.domain.packing_session import PackedSpan
from aiodoo_training.packing.token_rows import (
    emit_sequences,
    pad_row,
    resolve_token_rows,
    round_up,
)
from aiodoo_training.ports.packing import PackingStrategy
from aiodoo_training.registries import packing_registry


class NoPackingStrategy(PackingStrategy):
    """
    Passthrough packing — one padded row per example (O(n)).

    Does not concatenate. Used when PackingMode.NONE.
    """

    BACKEND_KEY = "none"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context

    def bind(self, context: Any) -> NoPackingStrategy:
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
        pad_mult = policy.pad_to_multiple_of if policy else None
        width = round_up(max_len, pad_mult)
        provided = dict(getattr(ctx, "token_rows", {}) or {})
        rows = resolve_token_rows(examples, max_length=max_len, provided=provided)

        sequences = []
        for example in examples:
            row = pad_row(rows[example.example_id], width=width)
            sequences.append(
                (
                    row.input_ids,
                    row.attention_mask,
                    row.labels,
                    (PackedSpan(example.example_id, 0, sum(row.attention_mask)),),
                    (example.example_id,),
                )
            )
        if not sequences:
            return
            yield  # pragma: no cover
        yield emit_sequences(sequences, width=width)


def register_default_packing(*, overwrite: bool = False) -> None:
    """Register none + Phase 5 packing strategies."""
    from aiodoo_training.packing.best_fit import register_best_fit_packing
    from aiodoo_training.packing.concat import register_concat_packing
    from aiodoo_training.packing.length_aware import register_length_aware_packing

    if not packing_registry.exists("none") or overwrite:
        packing_registry.register("none", NoPackingStrategy, overwrite=overwrite)
    register_concat_packing(overwrite=overwrite)
    register_best_fit_packing(overwrite=overwrite)
    register_length_aware_packing(overwrite=overwrite)
