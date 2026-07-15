"""Golden determinism tests for Phase 5 packing."""

from __future__ import annotations

import pytest

from aiodoo_training.bootstrap import bootstrap_phase5
from aiodoo_training.domain.enums import PackingMode
from tests.unit.phase5_helpers import plan_once


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase5(overwrite=True)


@pytest.mark.parametrize(
    "backend,mode",
    [
        ("none", PackingMode.NONE),
        ("concat", PackingMode.CONCAT),
        ("best_fit", PackingMode.BEST_FIT),
        ("length_aware", PackingMode.LENGTH_AWARE),
    ],
)
def test_golden_packing_identical(backend: str, mode: PackingMode) -> None:
    a = plan_once(packing_backend=backend, packing_mode=mode, seed=42)
    b = plan_once(packing_backend=backend, packing_mode=mode, seed=42)
    assert a.packing_fingerprint == b.packing_fingerprint
    assert a.packing_statistics == b.packing_statistics
    assert len(a.token_batches) == len(b.token_batches)
    for ba, bb in zip(a.token_batches, b.token_batches, strict=True):
        assert ba.example_ids == bb.example_ids
        assert ba.input_ids == bb.input_ids
        assert ba.attention_mask == bb.attention_mask
        assert ba.labels == bb.labels
        assert ba.metadata == bb.metadata
