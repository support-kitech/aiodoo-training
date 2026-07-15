"""Phase 4 evaluation convenience harness (CPU stub helpers)."""

from __future__ import annotations

from pathlib import Path

from aiodoo_training.builders.evaluation_builders import (
    EvaluationContextBuilder,
    enable_evaluation,
    make_evaluation_session,
)
from aiodoo_training.domain.config import ExperimentConfig
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.evaluation_policies import AcceptancePolicy, EvaluationPolicy
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.evaluation.context import EvaluationContext
from aiodoo_training.evaluation.engine import EvaluationEngine
from aiodoo_training.factories import EvaluatorFactory, RngControllerFactory
from aiodoo_training.training.engine import (
    build_stub_training_context,
    make_stub_experiment_config,
)


def _eval_ref(path: Path) -> DatasetRef:
    return DatasetRef(
        path=path,
        dataset_type=DatasetType.EVALUATION,
        protocol_version="1.0",
        name="stub-eval",
    )


def build_stub_evaluation_context(
    *,
    config: ExperimentConfig | None = None,
    output_dir: Path | None = None,
    dataset_path: Path | None = None,
    seed: int = 42,
    acceptance: AcceptancePolicy | None = None,
) -> EvaluationContext:
    """Build a bound stub EvaluationContext for CPU golden / unit tests."""
    if config is None:
        if output_dir is None:
            raise ValueError("output_dir required when config is omitted")
        config = make_stub_experiment_config(output_dir=output_dir, seed=seed)

    refs: tuple[DatasetRef, ...] = ()
    if dataset_path is not None:
        refs = (_eval_ref(dataset_path),)
    elif config.evaluation.dataset_refs:
        refs = config.evaluation.dataset_refs
    else:
        # Synthetic path used only as a deterministic identity key for stub metrics.
        refs = (_eval_ref(Path("fixture/eval.jsonl")),)

    config = enable_evaluation(config, dataset_refs=refs)
    train_ctx = build_stub_training_context(config=config)
    experiment_id = config.experiment_id or ExperimentId(value=config.name)
    run_id = RunId(value="eval-run")
    session = make_evaluation_session(
        experiment_id=experiment_id,
        run_id=run_id,
        model_fingerprint=train_ctx.model_fingerprint,
        adapter_fingerprint=train_ctx.adapter_fingerprint,
        config_fingerprint=train_ctx.config_fingerprint,
        execution_digest=train_ctx.execution_digest,
    )
    evaluator = EvaluatorFactory().create("stub")
    policy = EvaluationPolicy(
        backend_key="stub",
        seed=seed,
        metrics=("loss", "perplexity", "token_accuracy"),
    )
    rng = RngControllerFactory().create("python")
    ctx = (
        EvaluationContextBuilder()
        .with_config(config)
        .with_piece("execution", train_ctx.execution)
        .with_piece("model", train_ctx.model)
        .with_piece("evaluator", evaluator)
        .with_piece("evaluation_session", session)
        .with_piece("evaluation_policy", policy)
        .with_piece("acceptance_policy", acceptance or AcceptancePolicy())
        .with_piece("dataset_refs", refs)
        .with_piece("evaluator_backend_key", "stub")
        .with_piece("model_fingerprint", train_ctx.model_fingerprint)
        .with_piece("adapter_fingerprint", train_ctx.adapter_fingerprint)
        .with_piece("config_fingerprint", train_ctx.config_fingerprint)
        .with_piece("execution_digest", train_ctx.execution_digest)
        .with_piece("rng", rng)
        .build()
    )
    evaluator.bind(ctx)  # type: ignore[attr-defined]
    return ctx


def run_stub_evaluate(
    ctx: EvaluationContext | None = None,
    *,
    output_dir: Path | None = None,
) -> tuple[EvaluationContext, object]:
    """Run EvaluationEngine with stub evaluator; return context + report."""
    if ctx is None:
        if output_dir is None:
            raise ValueError("output_dir required when context is omitted")
        ctx = build_stub_evaluation_context(output_dir=output_dir)
    engine = EvaluationEngine()
    return engine.run(ctx)
