"""EvaluationEngine — prepare, evaluate via Evaluator port, collect metrics."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from aiodoo_training.domain.artifacts import EvaluationReport
from aiodoo_training.domain.enums import EvaluationStatus
from aiodoo_training.domain.identifiers import RunId
from aiodoo_training.domain.quality import QualityReport
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.evaluation.context import EvaluationContext
from aiodoo_training.evaluation.lifecycle import EvaluationLifecycle
from aiodoo_training.evaluation.metrics import MetricCollector, MetricHistory
from aiodoo_training.evaluation.quality_gate import QualityGate


class EvaluationEngine:
    """
    Application orchestrator for offline evaluation.

    Delegates metric computation to the bound :class:`Evaluator` port;
    owns session lifecycle and metric collection.
    """

    def __init__(
        self,
        *,
        lifecycle: EvaluationLifecycle | None = None,
        quality_gate: QualityGate | None = None,
    ) -> None:
        self._lifecycle = lifecycle or EvaluationLifecycle()
        self._quality_gate = quality_gate or QualityGate()

    def prepare(self, context: EvaluationContext) -> EvaluationContext:
        """Open dataset sessions and transition session to RUNNING or SKIPPED."""
        spec = context.evaluation_spec

        if not spec.enabled:
            session = self._lifecycle.skip(
                context.evaluation_session,
                message="evaluation disabled",
            )
            return context.with_evaluation_session(session)

        if not context.dataset_refs:
            session = self._lifecycle.skip(
                context.evaluation_session,
                message="no dataset refs",
            )
            return context.with_evaluation_session(session)

        sessions = self._open_dataset_sessions(context)
        session = context.evaluation_session
        if session.status == EvaluationStatus.PENDING:
            session = self._lifecycle.start(session)
        session = session.with_dataset_session(sessions[0] if sessions else None)
        return context.with_evaluation_session(session).with_dataset_sessions(sessions)

    def run(self, context: EvaluationContext) -> tuple[EvaluationContext, EvaluationReport]:
        """Evaluate via bound Evaluator and collect metrics."""
        ctx = self.prepare(context)
        session = ctx.evaluation_session

        if session.status == EvaluationStatus.SKIPPED:
            report = EvaluationReport(
                experiment_id=session.experiment_id,
                run_id=session.run_id,
                metrics=(),
                passed=True,
                details="evaluation skipped",
                created_at=datetime.now(UTC),
            )
            return ctx.with_evaluation_report(report), report

        seed = ctx.evaluation_policy.seed
        if seed is None:
            seed = ctx.config.seed
        rng = ctx.rng
        if rng is not None and hasattr(rng, "seed_all"):
            rng.seed_all(seed)

        evaluator = ctx.evaluator
        if hasattr(evaluator, "bind"):
            evaluator.bind(ctx)

        try:
            report = evaluator.evaluate(
                ctx.model,
                ctx.dataset_refs,
                ctx.evaluation_spec,
                session.experiment_id,
                session.run_id,
                ctx.execution,
            )
            session = self._lifecycle.complete(session)
        except Exception as exc:  # noqa: BLE001 — transition then re-raise
            session = self._lifecycle.fail(session, message=str(exc))
            ctx = ctx.with_evaluation_session(session)
            raise

        collector = MetricCollector(MetricHistory())
        collector.observe_many(report.metrics)

        report_id = f"eval-{uuid4().hex[:12]}"
        session = session.with_report(report_id=report_id)
        ctx = ctx.with_evaluation_session(session).with_evaluation_report(report)
        return ctx, report

    def validate_gates(
        self,
        report: EvaluationReport,
        context: EvaluationContext,
    ) -> QualityReport:
        """Run quality gates after evaluation report is finalized."""
        return self._quality_gate.validate(report, context.acceptance_policy)

    def _open_dataset_sessions(self, context: EvaluationContext) -> tuple[DatasetSession, ...]:
        if context.dataset_sessions:
            return context.dataset_sessions
        sessions: list[DatasetSession] = []
        for index, ref in enumerate(context.dataset_refs):
            sessions.append(
                DatasetSession(
                    session_id=f"eval-ds-{uuid4().hex[:8]}-{index}",
                    experiment_id=context.evaluation_session.experiment_id,
                    run_id=context.evaluation_session.run_id or RunId(value="eval-run"),
                    shuffle_seed=context.evaluation_policy.seed,
                    metadata={"dataset_path": str(ref.path), "split_index": str(index)},
                )
            )
        return tuple(sessions)
