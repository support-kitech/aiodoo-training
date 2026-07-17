"""Unit tests for canonical training / adapter naming."""

from __future__ import annotations

import pytest

from aiodoo_training.naming import (
    adapter_product_id,
    internal_id_for,
    normalize_training_id,
    resolve_public_training_id,
)


def test_normalize_semantic_and_legacy() -> None:
    assert normalize_training_id("coding") == "coding"
    assert normalize_training_id("EXP-0001") == "coding"


def test_adapter_product_id() -> None:
    assert adapter_product_id("coding") == "aiodoo-coding"
    assert adapter_product_id("EXP-0001") == "aiodoo-coding"


def test_internal_id_for_bookkeeping() -> None:
    assert internal_id_for("coding") == "EXP-0001"
    assert internal_id_for("planner") is None


def test_resolve_public_training_id_from_config() -> None:
    assert (
        resolve_public_training_id(
            {"experiment": {"id": "coding", "internal_id": "EXP-0001"}}
        )
        == "coding"
    )
    assert resolve_public_training_id({"name": "EXP-0001"}) == "coding"


def test_unknown_id_raises() -> None:
    with pytest.raises(ValueError):
        normalize_training_id("EXP-9999")
    with pytest.raises(ValueError):
        normalize_training_id("not-a-training")
