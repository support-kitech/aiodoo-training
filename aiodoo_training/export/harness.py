"""Phase 4 export convenience harness (CPU stub helpers)."""

from __future__ import annotations

from pathlib import Path

from aiodoo_training.builders.export_builders import ExportContextBuilder, make_export_session
from aiodoo_training.domain.artifacts import EvaluationReport
from aiodoo_training.domain.config import ExperimentConfig
from aiodoo_training.domain.export_manifest import ArtifactBundle, ArtifactCompatibilityPolicy
from aiodoo_training.domain.identifiers import RunId
from aiodoo_training.domain.quality import QualityReport
from aiodoo_training.export.context import ExportContext
from aiodoo_training.export.manager import ExportManager
from aiodoo_training.factories import ExporterFactory
from aiodoo_training.training.engine import (
    build_stub_training_context,
    make_stub_experiment_config,
)


def build_stub_export_context(
    *,
    config: ExperimentConfig | None = None,
    output_dir: Path,
    evaluation_report: EvaluationReport | None = None,
    quality_report: QualityReport | None = None,
    evaluation_fingerprint: str | None = None,
    require_evaluation: bool = False,
    require_pass_for_export: bool = False,
) -> ExportContext:
    """Build a bound stub ExportContext for CPU golden / unit tests."""
    if config is None:
        config = make_stub_experiment_config(output_dir=output_dir)

    train_ctx = build_stub_training_context(config=config)
    from aiodoo_training.domain.identifiers import ExperimentId

    experiment_id = config.experiment_id or ExperimentId(value=config.name)
    run_id = RunId(value="export-run")
    session = make_export_session(
        experiment_id=experiment_id,
        run_id=run_id,
        model_fingerprint=train_ctx.model_fingerprint,
        adapter_fingerprint=train_ctx.adapter_fingerprint,
        config_fingerprint=train_ctx.config_fingerprint,
    )
    exporter = ExporterFactory().create("stub")
    export_types = (
        "peft_adapter",
        "tokenizer",
        "manifest",
        "model_card",
        "bundle",
    )
    compatibility = ArtifactCompatibilityPolicy(
        accepted_artifact_protocols=("1",),
        required_roles=("peft_adapter", "manifest"),
        optional_roles=("tokenizer", "model_card", "evaluation_report"),
    )
    ctx = (
        ExportContextBuilder()
        .with_config(config)
        .with_piece("execution", train_ctx.execution)
        .with_piece("model", train_ctx.model)
        .with_piece("exporter", exporter)
        .with_piece("export_session", session)
        .with_piece("output_dir", output_dir)
        .with_piece("exporter_backend_key", "stub")
        .with_piece("model_fingerprint", train_ctx.model_fingerprint)
        .with_piece("adapter_fingerprint", train_ctx.adapter_fingerprint)
        .with_piece("config_fingerprint", train_ctx.config_fingerprint)
        .with_piece("evaluation_fingerprint", evaluation_fingerprint)
        .with_piece("evaluation_report", evaluation_report)
        .with_piece("quality_report", quality_report)
        .with_piece("export_types", export_types)
        .with_piece("compatibility_policy", compatibility)
        .with_piece("require_evaluation", require_evaluation)
        .with_piece("require_pass_for_export", require_pass_for_export)
        .build()
    )
    exporter.bind(ctx)  # type: ignore[attr-defined]
    return ctx


def run_stub_export(
    ctx: ExportContext | None = None,
    *,
    output_dir: Path | None = None,
) -> tuple[ExportContext, ArtifactBundle]:
    """Run ExportManager with stub exporter; return context + bundle."""
    if ctx is None:
        if output_dir is None:
            raise ValueError("output_dir required when context is omitted")
        ctx = build_stub_export_context(output_dir=output_dir)
    return ExportManager().export(ctx)
