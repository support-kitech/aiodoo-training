"""Phase 4 evaluation unit tests — lifecycle, engine, gates, metrics."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase4
from aiodoo_training.builders import EvaluationBuilder
from aiodoo_training.domain.artifacts import EvaluationReport
from aiodoo_training.domain.enums import ComparisonOp, EvaluationStatus
from aiodoo_training.domain.evaluation_policies import (
    AcceptancePolicy,
    QualityThreshold,
    ThresholdCombine,
    ThresholdSeverity,
)
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.training import MetricSnapshot
from aiodoo_training.evaluation import (
    EvaluationEngine,
    EvaluationLifecycle,
    MetricCollector,
    MetricHistory,
    QualityGate,
    build_stub_evaluation_context,
    run_stub_evaluate,
)
from aiodoo_training.exceptions import EvaluationLifecycleError
from aiodoo_training.factories import EvaluatorFactory
from aiodoo_training.infrastructure.stub.evaluator import StubEvaluator


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase4(overwrite=True)


def _session() -> object:
    from aiodoo_training.builders.evaluation_builders import make_evaluation_session

    return make_evaluation_session(
        experiment_id=ExperimentId(value="e1"),
        run_id=RunId(value="r1"),
    )


def test_evaluation_lifecycle_happy_path() -> None:
    life = EvaluationLifecycle()
    s = life.start(_session())
    assert s.status is EvaluationStatus.RUNNING
    s = life.complete(s)
    assert s.status is EvaluationStatus.COMPLETED


def test_evaluation_lifecycle_skip_and_fail() -> None:
    life = EvaluationLifecycle()
    s = life.skip(_session())
    assert s.status is EvaluationStatus.SKIPPED
    running = life.start(_session())
    failed = life.fail(running, message="boom")
    assert failed.status is EvaluationStatus.FAILED
    recovered = life.fresh_session(failed)
    assert recovered.status is EvaluationStatus.PENDING


def test_evaluation_lifecycle_rejects_illegal() -> None:
    life = EvaluationLifecycle()
    with pytest.raises(EvaluationLifecycleError):
        life.complete(_session())


def test_evaluation_engine_produces_core_metrics(tmp_path: Path) -> None:
    ctx = build_stub_evaluation_context(output_dir=tmp_path, seed=42)
    engine = EvaluationEngine()
    updated, report = engine.run(ctx)
    names = {m.name for m in report.metrics}
    assert names == {"loss", "perplexity", "token_accuracy"}
    assert updated.evaluation_session.status is EvaluationStatus.COMPLETED
    for metric in report.metrics:
        assert isinstance(metric.value, float)


def test_quality_gate_pass_and_fail(tmp_path: Path) -> None:
    _, report = run_stub_evaluate(output_dir=tmp_path)
    gate = QualityGate()
    loss = next(m.value for m in report.metrics if m.name == "loss")

    passing = AcceptancePolicy(
        thresholds=(
            QualityThreshold(
                metric_key="loss",
                op=ComparisonOp.LE,
                value=loss + 1.0,
            ),
            QualityThreshold(
                metric_key="token_accuracy",
                op=ComparisonOp.GE,
                value=0.0,
            ),
        ),
        combine=ThresholdCombine.ALL,
    )
    pass_report = gate.validate(report, passing)
    assert pass_report.passed

    failing = AcceptancePolicy(
        thresholds=(
            QualityThreshold(
                metric_key="loss",
                op=ComparisonOp.GE,
                value=loss + 100.0,
            ),
        ),
    )
    fail_report = gate.validate(report, failing)
    assert not fail_report.passed
    assert fail_report.failures
    assert fail_report.failures[0].metric_key == "loss"
    assert fail_report.failures[0].expected.startswith(">=")


def test_metric_collector_and_aggregator() -> None:
    ts = datetime(2025, 1, 1, tzinfo=UTC)
    snapshots = (
        MetricSnapshot(name="loss", value=1.0, step=1, timestamp=ts),
        MetricSnapshot(name="loss", value=3.0, step=2, timestamp=ts),
        MetricSnapshot(name="perplexity", value=2.5, step=1, timestamp=ts),
    )
    collector = MetricCollector(MetricHistory())
    collector.observe_many(snapshots)
    assert len(collector.history.snapshots) == 3
    assert collector.aggregator.mean("loss") == 2.0
    agg = collector.aggregator.aggregated_snapshots(step=99, timestamp=ts)
    assert {s.name for s in agg} == {"loss_mean", "perplexity_mean"}


def test_evaluator_factory_creates_stub() -> None:
    evaluator = EvaluatorFactory().create("stub")
    assert isinstance(evaluator, StubEvaluator)
    assert evaluator.BACKEND_KEY == "stub"


def test_evaluation_builder_builds_policy() -> None:
    acceptance = AcceptancePolicy(
        thresholds=(
            QualityThreshold(
                metric_key="loss",
                op=ComparisonOp.LE,
                value=2.0,
                severity=ThresholdSeverity.WARN,
            ),
        ),
    )
    policy = (
        EvaluationBuilder()
        .with_backend("stub")
        .with_profile("default")
        .with_metrics("loss", "perplexity")
        .with_seed(7)
        .with_acceptance(acceptance)
        .build_policy()
    )
    assert policy.backend_key == "stub"
    assert policy.metrics == ("loss", "perplexity")
    assert policy.seed == 7
    built_acceptance = EvaluationBuilder().with_acceptance(acceptance).build_acceptance()
    assert built_acceptance.thresholds[0].op is ComparisonOp.LE
    assert built_acceptance.thresholds[0].op.value == "<="


def test_bound_evaluator_returns_report_not_export(tmp_path: Path) -> None:
    ctx = build_stub_evaluation_context(output_dir=tmp_path)
    evaluator = ctx.evaluator
    report = evaluator.evaluate(
        ctx.model,
        ctx.dataset_refs,
        ctx.evaluation_spec,
        ctx.evaluation_session.experiment_id,
        ctx.evaluation_session.run_id,
        ctx.execution,
    )
    assert isinstance(report, EvaluationReport)
    assert not hasattr(report, "export_type")
    assert not hasattr(report, "root")
