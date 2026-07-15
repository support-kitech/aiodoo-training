"""Phase 5 configuration, registry, complexity, and statistics tests."""

from __future__ import annotations

import time

import pytest

from aiodoo_training.bootstrap import bootstrap_phase5
from aiodoo_training.config.curriculum_config import parse_curriculum_config, to_curriculum_spec
from aiodoo_training.config.packing_config import parse_packing_config, to_packing_policy
from aiodoo_training.config.sampling_config import parse_sampling_config, to_sampling_spec
from aiodoo_training.domain.enums import PackingMode
from aiodoo_training.exceptions import ConfigError
from aiodoo_training.registries import curriculum_registry, packing_registry, sampling_registry
from tests.unit.phase5_helpers import make_examples, plan_once


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase5(overwrite=True)


def test_parse_packing_config_defaults() -> None:
    frag = parse_packing_config({})
    policy = to_packing_policy(frag)
    assert policy.backend_key == "none"
    assert policy.mode is PackingMode.NONE


def test_parse_packing_config_rejects_empty_backend() -> None:
    with pytest.raises(ConfigError):
        parse_packing_config({"backend": ""})


def test_parse_curriculum_and_sampling() -> None:
    c = parse_curriculum_config(
        {"backend": "sequential", "mode": "sequential", "stages": ["a", "b"]}
    )
    spec = to_curriculum_spec(c)
    assert spec.stages == ("a", "b")
    s = parse_sampling_config({"backend": "temperature", "temperature": 0.7, "seed": 3})
    samp = to_sampling_spec(s)
    assert samp.temperature == 0.7
    assert samp.seed == 3


def test_registries_contain_phase5_keys() -> None:
    for key in ("none", "concat", "best_fit", "length_aware"):
        assert packing_registry.exists(key)
    for key in ("none", "sequential", "weighted", "difficulty", "random", "mixed"):
        assert curriculum_registry.exists(key)
    for key in ("identity", "weighted", "temperature", "balanced"):
        assert sampling_registry.exists(key)


def test_complexity_sanity_best_fit_not_quadratic() -> None:
    """Sanity: packing 400 examples with best_fit finishes quickly (O(n log n))."""
    examples = make_examples(400)
    start = time.perf_counter()
    plan = plan_once(
        examples,
        packing_backend="best_fit",
        packing_mode=PackingMode.BEST_FIT,
        max_sequence_length=128,
    )
    elapsed = time.perf_counter() - start
    assert plan.packing_statistics.examples_input == 400
    # Extremely loose CPU bound — guards against accidental O(n²) blowups.
    assert elapsed < 5.0


def test_statistics_equality_surface_no_timestamps() -> None:
    a = plan_once()
    b = plan_once()
    assert a.packing_statistics == b.packing_statistics
    assert a.curriculum_statistics == b.curriculum_statistics
