"""Seeded dataset mixing."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator, Mapping, Sequence

from aiodoo_training.domain.examples import TrainingExample


def mix_examples(
    groups: Sequence[tuple[TrainingExample, ...]],
    *,
    weights: Sequence[float] | None = None,
    shuffle: bool = True,
    seed: int = 42,
    cap: int | None = None,
) -> tuple[TrainingExample, ...]:
    """
    Merge example groups into a deterministic ordered tuple.

    When ``shuffle`` is True, examples are shuffled with a seeded RNG.
    When weights are provided (same length as groups), groups are expanded by
    relative weight via proportional repetition before shuffle (deterministic).
    """
    if not groups:
        return ()

    if weights is None or len({float(w) for w in weights}) == 1:
        pooled = [ex for group in groups for ex in group]
    else:
        if len(weights) != len(groups):
            raise ValueError("weights length must match groups length.")
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative.")
        positive = [w for w in weights if w > 0]
        if not positive:
            raise ValueError("weights must include at least one positive value.")
        min_w = min(positive)
        pooled = []
        for group, weight in zip(groups, weights, strict=True):
            if weight <= 0 or not group:
                continue
            repeats = max(1, int(round(weight / min_w)))
            for _ in range(repeats):
                pooled.extend(group)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pooled)

    if cap is not None:
        if cap < 0:
            raise ValueError("cap must be >= 0.")
        pooled = pooled[:cap]
    return tuple(pooled)


def stable_example_id(dataset_type: str, record: Mapping[str, object], index: int) -> str:
    """Derive a stable example id from record fields or content hash."""
    for key in ("id", "review_id", "evaluation_id", "example_id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return f"{dataset_type}:{value.strip()}"
    payload = repr(sorted(record.items())).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{dataset_type}:{index}:{digest}"


def iter_mixed(
    examples: Sequence[TrainingExample],
) -> Iterator[TrainingExample]:
    """Iterate a materialized mix tuple."""
    yield from examples
