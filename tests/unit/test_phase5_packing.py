"""Unit tests for Phase 5 packing strategies, sessions, and planner."""

from __future__ import annotations

import pytest

from aiodoo_training.bootstrap import bootstrap_phase5
from aiodoo_training.domain.enums import PackingMode, PackingOverflow, PackingStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.packing_policies import PackingPolicy
from aiodoo_training.domain.packing_session import PackingSession
from aiodoo_training.exceptions import PackingError, PackingLifecycleError
from aiodoo_training.factories import PackingStrategyFactory
from aiodoo_training.packing.lifecycle import PackingLifecycle
from tests.unit.phase5_helpers import make_examples, plan_once


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase5(overwrite=True)


def test_packing_registry_keys() -> None:
    factory = PackingStrategyFactory()
    for key in ("none", "concat", "best_fit", "length_aware"):
        assert factory.create(key) is not None


def test_no_packing_emits_one_row_per_example() -> None:
    plan = plan_once(packing_backend="none", packing_mode=PackingMode.NONE)
    assert plan.packing_statistics.examples_input == 5
    assert plan.packing_statistics.sequences_emitted == 5


def test_concat_packing_reduces_sequences() -> None:
    plan = plan_once(packing_backend="concat", packing_mode=PackingMode.CONCAT)
    assert plan.packing_statistics.sequences_emitted <= 5
    assert plan.packing_statistics.sequences_emitted >= 1
    batch = plan.token_batches[0]
    assert batch.metadata is not None
    assert "packed_spans" in batch.metadata


def test_best_fit_and_length_aware_produce_batches() -> None:
    for backend, mode in (
        ("best_fit", PackingMode.BEST_FIT),
        ("length_aware", PackingMode.LENGTH_AWARE),
    ):
        plan = plan_once(packing_backend=backend, packing_mode=mode)
        assert len(plan.token_batches) == 1
        assert plan.packing_statistics.examples_packed >= 1


def test_reject_overflow_raises() -> None:
    from aiodoo_training.domain.config import PackingSpec
    from aiodoo_training.domain.identifiers import ExperimentId, RunId
    from aiodoo_training.packing.context import PackingContext
    from aiodoo_training.packing.token_rows import resolve_token_rows

    examples = make_examples(1)
    # Force a tiny max length with reject overflow.
    policy = PackingPolicy(
        backend_key="concat",
        mode=PackingMode.CONCAT,
        max_sequence_length=2,
        overflow=PackingOverflow.REJECT,
    )
    session = PackingSession(
        session_id="p",
        experiment_id=ExperimentId(value="e"),
        run_id=RunId(value="r"),
    )
    # Stub row longer than max via provided map after bind would still truncate
    # in resolve — use concat with short max and long content already.
    rows = resolve_token_rows(examples, max_length=10_000)
    # Manually elongate
    long = rows[examples[0].example_id]
    from aiodoo_training.packing.token_rows import TokenRow

    elongated = TokenRow(
        input_ids=long.input_ids + (1,) * 50,
        attention_mask=long.attention_mask + (1,) * 50,
        labels=long.labels + (1,) * 50,
    )
    ctx = PackingContext(
        examples=examples,
        packing_session=session,
        packing_spec=PackingSpec(mode=PackingMode.CONCAT, max_sequence_length=2),
        packing_policy=policy,
        token_rows={examples[0].example_id: elongated},
    )
    strategy = PackingStrategyFactory().create("concat")
    strategy.bind(ctx)  # type: ignore[attr-defined]
    with pytest.raises(PackingError):
        list(strategy.pack(examples, PackingSpec(mode=PackingMode.CONCAT, max_sequence_length=2)))


def test_packing_lifecycle_transitions() -> None:
    life = PackingLifecycle()
    session = PackingSession(
        session_id="p1",
        experiment_id=ExperimentId(value="e"),
        run_id=RunId(value="r"),
    )
    planning = life.begin(session)
    assert planning.status is PackingStatus.PLANNING
    ready = life.ready(planning)
    assert ready.status is PackingStatus.READY
    with pytest.raises(PackingLifecycleError):
        life.begin(ready)


def test_packing_statistics_immutable_equality() -> None:
    a = plan_once()
    b = plan_once()
    assert a.packing_statistics == b.packing_statistics
    assert a.packing_fingerprint == b.packing_fingerprint
