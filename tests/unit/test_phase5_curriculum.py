"""Unit tests for Phase 5 curriculum strategies."""

from __future__ import annotations

import pytest

from aiodoo_training.bootstrap import bootstrap_phase5
from aiodoo_training.curriculum import CurriculumLifecycle
from aiodoo_training.domain.curriculum_session import CurriculumSession
from aiodoo_training.domain.enums import CurriculumMode, CurriculumStatus, PackingMode
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.exceptions import CurriculumLifecycleError
from aiodoo_training.factories import CurriculumStrategyFactory
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
def test_curriculum_backends_plan(backend: str, mode: CurriculumMode) -> None:
    plan = plan_once(
        curriculum_backend=backend,
        curriculum_mode=mode,
        packing_backend="none",
        packing_mode=PackingMode.NONE,
    )
    assert plan.curriculum_statistics.examples_total == 5
    assert plan.curriculum_statistics.stage_count >= 1
    ids = [e.example_id for e in plan.ordered_examples]
    assert len(ids) == 5
    assert set(ids) == {f"e{i}" for i in range(5)}


def test_difficulty_orders_ascending() -> None:
    plan = plan_once(
        curriculum_backend="difficulty",
        curriculum_mode=CurriculumMode.DIFFICULTY,
        stages=("low", "high"),
        packing_backend="none",
        packing_mode=PackingMode.NONE,
    )
    ids = [e.example_id for e in plan.ordered_examples]
    assert ids == ["e0", "e1", "e2", "e3", "e4"]


def test_curriculum_lifecycle() -> None:
    life = CurriculumLifecycle()
    session = CurriculumSession(
        session_id="c1",
        experiment_id=ExperimentId(value="e"),
        run_id=RunId(value="r"),
    )
    planning = life.begin(session)
    ready = life.ready(planning)
    active = life.activate(ready)
    done = life.complete(active)
    assert done.status is CurriculumStatus.COMPLETED
    with pytest.raises(CurriculumLifecycleError):
        life.begin(done)


def test_curriculum_statistics_equality() -> None:
    a = plan_once(curriculum_backend="sequential")
    b = plan_once(curriculum_backend="sequential")
    assert a.curriculum_statistics == b.curriculum_statistics
    assert a.curriculum_fingerprint == b.curriculum_fingerprint


def test_curriculum_registry() -> None:
    factory = CurriculumStrategyFactory()
    for key in ("none", "sequential", "weighted", "difficulty", "random", "mixed"):
        assert factory.create(key) is not None
