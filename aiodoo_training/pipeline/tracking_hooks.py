"""Pipeline helpers for observational Phase 6 tracking (non-authoritative)."""

from __future__ import annotations

from pathlib import Path

from aiodoo_training.builders.tracking_builders import TrackingBuilder
from aiodoo_training.config.tracking_config import parse_tracking_config, to_tracking_policy
from aiodoo_training.domain.enums import RunState
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.tracking_reports import (
    EvaluationReportSummary,
    ExportReport,
    TrainingReport,
)
from aiodoo_training.domain.training import MetricSnapshot
from aiodoo_training.factories.factories import TrackerFactory
from aiodoo_training.pipeline.pipeline import PipelineContext
from aiodoo_training.tracking.core import TrackingCoordinator, new_run_record
from aiodoo_training.tracking.reports import write_run_reports


def maybe_open_tracking(context: PipelineContext) -> PipelineContext:
    """Open TrackingCoordinator if enabled in raw_config; never fails the pipeline."""
    if context.get("tracking_coordinator") is not None:
        return context
    config = context.config
    if config is None:
        return context
    raw = context.get("raw_config") or {}
    tracking_raw = raw.get("tracking") if isinstance(raw, dict) else None
    fragment = parse_tracking_config(tracking_raw if isinstance(tracking_raw, dict) else {})
    policy = to_tracking_policy(fragment)
    if not policy.enabled:
        return context

    experiment_id = (
        context.experiment_id or config.experiment_id or ExperimentId(value=config.name)
    )
    run_id = context.run_id or RunId(value="run")
    root = policy.root_dir or Path("artifacts/tracking")

    builder = TrackingBuilder().with_policy(policy).with_root(root).with_identity(
        experiment_id=experiment_id,
        name=policy.experiment_name or config.name,
        config_fingerprint=str(context.get("config_fingerprint") or ""),
        model_fingerprint=str(context.get("model_fingerprint") or ""),
        adapter_fingerprint=str(context.get("adapter_fingerprint") or ""),
        execution_digest=str(context.get("execution_digest") or ""),
    )
    ctx = builder.build_context(
        run_record=new_run_record(experiment_id=experiment_id, run_id=run_id)
    )
    tracker = TrackerFactory().create(policy.backend_key)
    store = builder.build_store(root)
    coordinator = TrackingCoordinator(tracker=tracker, context=ctx, store=store)
    try:
        coordinator.open()
    except Exception:  # noqa: BLE001 — observational
        if not policy.nonfatal_sink_errors:
            raise
    return context.with_values(
        tracking_coordinator=coordinator,
        tracker=tracker,
        tracking_context=coordinator.context,
    )


def maybe_observe_progress(context: PipelineContext) -> None:
    coordinator = context.get("tracking_coordinator")
    progress = context.get("training_progress")
    if coordinator is None or progress is None:
        return
    metrics = getattr(progress, "metrics", ()) or ()
    if metrics:
        coordinator.observe_metrics(tuple(metrics))
    plan = context.get("schedule_plan")
    if plan is not None:
        try:
            coordinator.observe_statistics_blob(
                "packing_statistics", plan.packing_statistics
            )
            coordinator.observe_statistics_blob(
                "curriculum_statistics", plan.curriculum_statistics
            )
            run = coordinator.context.run_record.with_fingerprints(
                packing_fingerprint=plan.packing_fingerprint,
                curriculum_fingerprint=plan.curriculum_fingerprint,
            )
            coordinator._ctx = coordinator.context.with_run(run)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass


def maybe_observe_evaluation(context: PipelineContext) -> None:
    coordinator = context.get("tracking_coordinator")
    report = context.get("evaluation_report")
    if coordinator is None or report is None:
        return
    metrics = tuple(
        MetricSnapshot(name=m.name, value=float(m.value), step=0)
        for m in getattr(report, "metrics", ())
    )
    if metrics:
        coordinator.observe_metrics(metrics)
    try:
        summary = EvaluationReportSummary(
            experiment_id=coordinator.context.experiment_session.experiment_id,
            run_id=coordinator.context.run_record.run_id,
            metric_names=tuple(m.name for m in metrics),
            metric_values=tuple(m.value for m in metrics),
        )
        if coordinator.context.report_policy.write_json:
            write_run_reports(
                run_dir=coordinator._run_root,  # noqa: SLF001
                evaluation=summary,
            )
    except Exception:  # noqa: BLE001
        pass


def maybe_observe_export(context: PipelineContext) -> None:
    coordinator = context.get("tracking_coordinator")
    bundle = context.get("artifact_bundle")
    if coordinator is None or bundle is None:
        return
    root = getattr(bundle, "root", None)
    if root is not None:
        coordinator.observe_artifact(Path(root), role="bundle")
    try:
        write_run_reports(
            run_dir=coordinator._run_root,  # noqa: SLF001
            export=ExportReport(
                experiment_id=coordinator.context.experiment_session.experiment_id,
                run_id=coordinator.context.run_record.run_id,
                artifact_refs=(str(root),) if root is not None else (),
                export_status="published",
            ),
        )
    except Exception:  # noqa: BLE001
        pass


def maybe_finalize_tracking(context: PipelineContext) -> None:
    coordinator = context.get("tracking_coordinator")
    if coordinator is None:
        return
    progress = context.get("training_progress")
    try:
        if progress is not None and coordinator.context.report_policy.write_json:
            write_run_reports(
                run_dir=coordinator._run_root,  # noqa: SLF001
                training=TrainingReport(
                    experiment_id=coordinator.context.experiment_session.experiment_id,
                    run_id=coordinator.context.run_record.run_id,
                    global_step=int(getattr(progress, "global_step", 0)),
                    epoch=float(getattr(progress, "epoch", 0.0)),
                    status=str(getattr(getattr(progress, "status", None), "value", "")),
                    metrics=tuple(getattr(progress, "metrics", ()) or ()),
                ),
            )
        status = getattr(getattr(progress, "status", None), "value", "")
        state = RunState.COMPLETED
        if status == "failed":
            state = RunState.FAILED
        coordinator.complete(state)
    except Exception:  # noqa: BLE001
        try:
            coordinator.close()
        except Exception:  # noqa: BLE001
            pass
