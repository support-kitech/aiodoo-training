"""Golden determinism tests for Phase 5 sampling."""

from __future__ import annotations

import pytest

from aiodoo_training.bootstrap import bootstrap_phase5
from aiodoo_training.domain.enums import PackingMode
from tests.unit.phase5_helpers import plan_once


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase5(overwrite=True)


@pytest.mark.parametrize("backend", ["identity", "weighted", "temperature", "balanced"])
def test_golden_sampling_identical(backend: str) -> None:
    a = plan_once(
        sampling_backend=backend,
        packing_backend="none",
        packing_mode=PackingMode.NONE,
        seed=123,
    )
    b = plan_once(
        sampling_backend=backend,
        packing_backend="none",
        packing_mode=PackingMode.NONE,
        seed=123,
    )
    assert [e.example_id for e in a.ordered_examples] == [
        e.example_id for e in b.ordered_examples
    ]
    assert a.sampling_fingerprint == b.sampling_fingerprint
