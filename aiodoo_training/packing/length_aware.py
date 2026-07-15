"""Length-aware packing — O(n log n) sort then concat."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from aiodoo_training.domain.config import PackingSpec
from aiodoo_training.domain.examples import TokenBatch, TrainingExample
from aiodoo_training.packing.concat import ConcatenationPacking
from aiodoo_training.packing.token_rows import resolve_token_rows
from aiodoo_training.ports.packing import PackingStrategy


class LengthAwarePacking(PackingStrategy):
    """Sort by token length (asc), then greedy concatenation."""

    BACKEND_KEY = "length_aware"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context
        self._concat = ConcatenationPacking(context)

    def bind(self, context: Any) -> LengthAwarePacking:
        self._context = context
        self._concat.bind(context)
        return self

    def pack(
        self,
        examples: Sequence[TrainingExample],
        spec: PackingSpec,
    ) -> Iterator[TokenBatch]:
        ctx = self._context
        policy = getattr(ctx, "packing_policy", None)
        max_len = int(policy.max_sequence_length if policy else spec.max_sequence_length)
        provided = dict(getattr(ctx, "token_rows", {}) or {})
        rows = resolve_token_rows(examples, max_length=max_len, provided=provided)
        ordered = sorted(examples, key=lambda e: (rows[e.example_id].length, e.example_id))
        yield from self._concat.pack(ordered, spec)
        # Propagate overflow counters from the concat binder back to this strategy.
        self._context = self._concat._context


def register_length_aware_packing(*, overwrite: bool = False) -> None:
    from aiodoo_training.registries import packing_registry

    if not packing_registry.exists("length_aware") or overwrite:
        packing_registry.register("length_aware", LengthAwarePacking, overwrite=overwrite)
