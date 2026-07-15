"""Unit tests for Phase 5 sampling strategies."""

from __future__ import annotations

import pytest

from aiodoo_training.bootstrap import bootstrap_phase5
from aiodoo_training.domain.enums import PackingMode
from aiodoo_training.domain.packing_policies import SamplingSpec
from aiodoo_training.factories import SamplingStrategyFactory
from aiodoo_training.sampling import (
    BalancedSampling,
    IdentitySampling,
    TemperatureSampling,
    WeightedSampling,
)
from tests.unit.phase5_helpers import make_examples, plan_once


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase5(overwrite=True)


def test_sampling_registry() -> None:
    factory = SamplingStrategyFactory()
    for key in ("identity", "weighted", "temperature", "balanced"):
        assert factory.create(key) is not None


def test_identity_preserves_order() -> None:
    examples = make_examples(4)
    out = IdentitySampling().sample(examples, SamplingSpec())
    assert [e.example_id for e in out] == [e.example_id for e in examples]


def test_weighted_and_temperature_deterministic() -> None:
    examples = make_examples(5)
    spec = SamplingSpec(seed=7, temperature=0.5)
    w1 = WeightedSampling().sample(examples, spec)
    w2 = WeightedSampling().sample(examples, spec)
    assert [e.example_id for e in w1] == [e.example_id for e in w2]
    t1 = TemperatureSampling().sample(examples, spec)
    t2 = TemperatureSampling().sample(examples, spec)
    assert [e.example_id for e in t1] == [e.example_id for e in t2]


def test_balanced_interleaves_strata() -> None:
    examples = make_examples(4)
    out = BalancedSampling().sample(
        examples, SamplingSpec(strata_key="dataset_type", seed=1)
    )
    types = [e.dataset_type for e in out]
    # Round-robin across sorted strata keys → alternating when counts equal.
    assert len(types) == 4
    assert set(e.example_id for e in out) == {e.example_id for e in examples}


def test_sampling_in_planner() -> None:
    a = plan_once(
        sampling_backend="weighted",
        packing_backend="none",
        packing_mode=PackingMode.NONE,
        seed=99,
    )
    b = plan_once(
        sampling_backend="weighted",
        packing_backend="none",
        packing_mode=PackingMode.NONE,
        seed=99,
    )
    assert [e.example_id for e in a.ordered_examples] == [
        e.example_id for e in b.ordered_examples
    ]
    assert a.sampling_fingerprint == b.sampling_fingerprint
