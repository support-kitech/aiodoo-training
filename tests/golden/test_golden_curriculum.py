"""Golden determinism tests for Phase 5 curriculum."""

from __future__ import annotations

import pytest

from aiodoo_training.bootstrap import bootstrap_phase5
from aiodoo_training.domain.enums import CurriculumMode, PackingMode
from tests.unit.phase5_helpers import plan_once


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase5(overwrite=True)


@pytest.mark.parametrize(
    "backend,mode",
    [
        ("none", CurriculumMode.NONE),
        ("sequential", CurriculumMode.SEQUENTIAL),
        ("weighted", CurriculumMode.WEIGHTED_MIX),
        ("difficulty", CurriculumMode.DIFFICULTY),
        ("random", CurriculumMode.RANDOM),
        ("mixed", CurriculumMode.MIXED),
    ],
)
def test_golden_curriculum_identical(backend: str, mode: CurriculumMode) -> None:
    a = plan_once(
        curriculum_backend=backend,
        curriculum_mode=mode,
        packing_backend="none",
        packing_mode=PackingMode.NONE,
        seed=42,
    )
    b = plan_once(
        curriculum_backend=backend,
        curriculum_mode=mode,
        packing_backend="none",
        packing_mode=PackingMode.NONE,
        seed=42,
    )
    assert [e.example_id for e in a.ordered_examples] == [
        e.example_id for e in b.ordered_examples
    ]
    assert [[e.example_id for e in stage] for stage in a.curriculum_stages] == [
        [e.example_id for e in stage] for stage in b.curriculum_stages
    ]
    assert a.curriculum_fingerprint == b.curriculum_fingerprint
    assert a.curriculum_statistics == b.curriculum_statistics
