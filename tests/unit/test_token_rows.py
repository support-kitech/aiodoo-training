"""Unit tests for token row helpers."""

from __future__ import annotations

from types import MappingProxyType

from aiodoo_training.domain.examples import IGNORE_INDEX, TokenBatch
from aiodoo_training.packing.token_rows import TokenRow, token_batch_to_rows


def test_token_batch_to_rows_preserves_hf_ids() -> None:
    batch = TokenBatch(
        example_ids=("ex-a", "ex-b"),
        input_ids=((101, 202, 303), (404, 505, 0)),
        attention_mask=((1, 1, 1), (1, 1, 0)),
        labels=((101, IGNORE_INDEX, 303), (404, 505, IGNORE_INDEX)),
    )
    rows = token_batch_to_rows(batch)
    assert set(rows) == {"ex-a", "ex-b"}
    assert rows["ex-a"].input_ids == (101, 202, 303)
    assert rows["ex-a"].attention_mask == (1, 1, 1)
    assert rows["ex-a"].labels == (101, IGNORE_INDEX, 303)
    assert rows["ex-b"].input_ids == (404, 505)
    assert rows["ex-b"].attention_mask == (1, 1)
    assert rows["ex-b"].labels == (404, 505)


def test_token_batch_to_rows_trims_trailing_pad() -> None:
    batch = TokenBatch(
        example_ids=("ex-a",),
        input_ids=((11, 22, 33, 0, 0),),
        attention_mask=((1, 1, 1, 0, 0),),
        labels=((11, 22, 33, IGNORE_INDEX, IGNORE_INDEX),),
    )
    row = token_batch_to_rows(batch)["ex-a"]
    assert row.input_ids == (11, 22, 33)
    assert row.attention_mask == (1, 1, 1)
    assert row.labels == (11, 22, 33)


def test_token_batch_to_rows_empty_batch() -> None:
    batch = TokenBatch(
        example_ids=(),
        input_ids=(),
        attention_mask=(),
        labels=(),
        metadata=MappingProxyType({}),
    )
    assert token_batch_to_rows(batch) == {}


def test_token_batch_to_rows_produces_token_row_compatible_with_packing() -> None:
    batch = TokenBatch(
        example_ids=("ex-a",),
        input_ids=((7, 8, 9),),
        attention_mask=((1, 1, 1),),
        labels=((7, 8, 9),),
    )
    row = token_batch_to_rows(batch)["ex-a"]
    assert isinstance(row, TokenRow)
    assert row.length == 3
