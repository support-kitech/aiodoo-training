"""Label masking and tokenization helpers."""

from __future__ import annotations

from aiodoo_training.domain.examples import IGNORE_INDEX


def apply_assistant_only_mask(
    full_ids: list[int],
    prompt_ids: list[int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> list[int]:
    """
    Build labels where the prompt prefix is ignored and assistant tokens kept.

    Assumes ``prompt_ids`` is a prefix of ``full_ids`` (common chat encoding).
    If the prefix relation fails, falls back to masking everything except tokens
    after ``len(prompt_ids)`` clipped to the full length.
    """
    labels = list(full_ids)
    prefix = min(len(prompt_ids), len(full_ids))
    if full_ids[:prefix] != prompt_ids[:prefix]:
        prefix = min(len(prompt_ids), len(full_ids))
    for i in range(prefix):
        labels[i] = ignore_index
    return labels


def pad_sequences(
    sequences: list[list[int]],
    *,
    max_length: int,
    pad_id: int,
    truncate: bool,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Truncate/pad sequences to ``max_length``.

    Returns (padded_ids, attention_masks).
    """
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    for seq in sequences:
        ids = list(seq)
        if truncate and len(ids) > max_length:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if len(ids) < max_length:
            pad_n = max_length - len(ids)
            ids = ids + [pad_id] * pad_n
            mask = mask + [0] * pad_n
        padded.append(ids)
        masks.append(mask)
    return padded, masks
