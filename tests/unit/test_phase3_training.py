"""Phase 3 unit tests — lifecycle, policies, events, metrics, factories."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aiodoo_training.bootstrap import bootstrap_phase3
from aiodoo_training.builders import TrainingContextBuilder
from aiodoo_training.config import (
    parse_resume_config,
    to_resume_policy,
    validate_phase3_fragments,
)
from aiodoo_training.domain.enums import TrainingStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.training import MetricSnapshot
from aiodoo_training.domain.training_events import TrainingEvent, TrainingEventKind
from aiodoo_training.domain.training_policies import TRAINING_PROTOCOL_VERSION, ResumePolicy
from aiodoo_training.domain.training_session import TrainingSession
from aiodoo_training.exceptions import BuilderError, TrainingLifecycleError
from aiodoo_training.factories import (
    CallbackFactory,
    CheckpointStoreFactory,
    OptimizerBackendFactory,
    RngControllerFactory,
    SchedulerBackendFactory,
    TrainerBackendFactory,
)
from aiodoo_training.ports.callback import TrainingCallback
from aiodoo_training.training import (
    CallbackContext,
    MetricAggregator,
    MetricCollector,
    TrainingEventBus,
    TrainingHistory,
    TrainingLifecycle,
)


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase3(overwrite=True)


def _session(status: TrainingStatus = TrainingStatus.PENDING) -> TrainingSession:
    return TrainingSession(
        session_id="s1",
        experiment_id=ExperimentId(value="e1"),
        run_id=RunId(value="r1"),
        status=status,
    )


def test_lifecycle_happy_path() -> None:
    life = TrainingLifecycle()
    s = life.start(_session())
    assert s.status is TrainingStatus.RUNNING
    s = life.pause(s)
    assert s.status is TrainingStatus.PAUSED
    s = life.resume_running(s)
    s = life.complete(s)
    assert s.status is TrainingStatus.COMPLETED


def test_lifecycle_rejects_illegal() -> None:
    life = TrainingLifecycle()
    with pytest.raises(TrainingLifecycleError):
        life.complete(_session())


def test_resume_policy_config_default_strict() -> None:
    frag = parse_resume_config({})
    assert to_resume_policy(frag) is ResumePolicy.STRICT
    bag = validate_phase3_fragments({"resume": {"policy": "warn"}})
    assert bag["resume_policy"] is ResumePolicy.WARN


def test_protocol_version_constant() -> None:
    assert TRAINING_PROTOCOL_VERSION == "1"


def test_factories_create_stub_backends() -> None:
    assert TrainerBackendFactory().create("stub").BACKEND_KEY == "stub"  # type: ignore[attr-defined]
    assert CheckpointStoreFactory().create("stub").BACKEND_KEY == "stub"  # type: ignore[attr-defined]
    rng = RngControllerFactory().create("python")
    rng.seed_all(7)
    snap = rng.snapshot()
    rng.seed_all(99)
    rng.restore(snap)
    assert rng.snapshot()["seed"] == 7
    OptimizerBackendFactory().create("adamw")
    SchedulerBackendFactory().create("cosine")
    CallbackFactory().create("null")


def test_event_bus_and_metrics() -> None:
    seen: list[TrainingEventKind] = []

    class Capture(TrainingCallback):
        def on_event(self, event: TrainingEvent, context: CallbackContext) -> None:
            seen.append(event.kind)

    history = TrainingHistory()
    collector = MetricCollector(history=history)
    # Minimal fake TrainingContext attributes for bus
    import tempfile
    from pathlib import Path

    from aiodoo_training.training.engine import (
        build_stub_training_context,
        make_stub_experiment_config,
    )

    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_stub_experiment_config(output_dir=Path(tmp), max_steps=2, save_steps=100)
        ctx = build_stub_training_context(config=cfg)
        ctx = ctx.__class__(
            **{
                **{f.name: getattr(ctx, f.name) for f in ctx.__dataclass_fields__.values()},  # type: ignore[attr-defined]
                "metric_collector": collector,
                "training_history": history,
            }
        )
        bus = TrainingEventBus()
        bus.subscribe(Capture())
        event = TrainingEvent(
            kind=TrainingEventKind.LOSS_COMPUTED,
            experiment_id=ctx.training_session.experiment_id,
            run_id=ctx.training_session.run_id,
            session_id=ctx.training_session.session_id,
            timestamp=datetime.now(UTC),
            global_step=1,
            loss=0.5,
        )
        bus.publish(event, ctx)
        assert seen == [TrainingEventKind.LOSS_COMPUTED]
        assert collector.history.snapshots[-1].value == 0.5


def test_metric_aggregator_mean() -> None:
    agg = MetricAggregator()
    agg.observe(MetricSnapshot(name="loss", value=1.0, step=1))
    agg.observe(MetricSnapshot(name="loss", value=3.0, step=2))
    assert agg.mean("loss") == 2.0


def test_training_context_builder_requires_pieces() -> None:
    with pytest.raises(BuilderError, match="missing required"):
        TrainingContextBuilder().with_config(
            # minimal fake — builder checks type only after pieces; use real config from helper
            __import__(
                "aiodoo_training.training.engine", fromlist=["make_stub_experiment_config"]
            ).make_stub_experiment_config(
                output_dir=__import__("pathlib").Path("/tmp/x"),
            )
        ).build()
