"""Convenience harness for Phase 3 stub training / resume (application helpers)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from aiodoo_training.adaptation import AdaptationApplier
from aiodoo_training.builders import AdaptationBuilder, ExecutionContextBuilder, ModelBuilder
from aiodoo_training.domain.config import (
    AdaptationSpec,
    CheckpointingSpec,
    CurriculumSpec,
    DatasetMixSpec,
    DeterminismSpec,
    EvaluationSpec,
    ExperimentConfig,
    ExportSpec,
    OptimizationSpec,
    PackingSpec,
    PrecisionSpec,
    TrackingSpec,
)
from aiodoo_training.domain.enums import AdapterType, ModelFamily, Precision, TrainingStatus
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.refs import ModelRef
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.domain.training import TrainingProgress
from aiodoo_training.domain.training_policies import CheckpointPolicy, ResumePolicy
from aiodoo_training.domain.training_session import TrainingSession
from aiodoo_training.factories import (
    AdaptationStrategyFactory,
    CheckpointStoreFactory,
    ModelBackendFactory,
    ResourcePlannerFactory,
    RngControllerFactory,
    TrainerBackendFactory,
)
from aiodoo_training.infrastructure.stub.trainer import StubTrainerBackend
from aiodoo_training.models import ModelLoader
from aiodoo_training.training.checkpoint_manager import (
    CheckpointManager,
    ResumeValidationContext,
)
from aiodoo_training.training.context import TrainingContext
from aiodoo_training.training.event_bus import TrainingEventBus
from aiodoo_training.training.metrics import MetricCollector, TrainingHistory
from aiodoo_training.training.resume import ResumeBundle, ResumeCoordinator


def make_stub_experiment_config(
    *,
    name: str = "stub-phase3",
    seed: int = 42,
    max_steps: int = 10,
    save_steps: int = 5,
    output_dir: Path,
    resume_from: Path | None = None,
    learning_rate: float = 2e-4,
) -> ExperimentConfig:
    """Build a minimal immutable ExperimentConfig for CPU stub training."""
    return ExperimentConfig(
        name=name,
        schema_version="1",
        seed=seed,
        model=ModelRef(
            identifier="fixture/stub-lm", family=ModelFamily.QWEN, precision=Precision.FP32
        ),
        datasets=DatasetMixSpec(),
        adaptation=AdaptationSpec(
            adapter_type=AdapterType.LORA, rank=8, alpha=16, target_modules=("q_proj",)
        ),
        optimization=OptimizationSpec(
            learning_rate=learning_rate,
            max_steps=max_steps,
            num_epochs=1.0,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
        ),
        precision=PrecisionSpec(precision=Precision.FP32),
        packing=PackingSpec(),
        curriculum=CurriculumSpec(),
        checkpointing=CheckpointingSpec(
            output_dir=output_dir,
            save_steps=save_steps,
            save_total_limit=5,
            resume_from=resume_from,
        ),
        evaluation=EvaluationSpec(),
        export=ExportSpec(),
        tracking=TrackingSpec(),
        determinism=DeterminismSpec(seed=seed),
        execution=ExecutionContextBuilder().with_device("cpu").build_spec(),
        experiment_id=ExperimentId(value=name),
        metadata=MappingProxyType({"training": {"backend": "stub"}}),
    )


def prepare_stub_trainable(
    *,
    seed: int = 42,
) -> tuple[TrainableModelHandle, ExecutionEnvironment, str, str]:
    """Load stub base model + LoRA adaptation; returns (model, execution, fingerprints)."""
    _ = seed
    planner = ResourcePlannerFactory().create("static")
    execution = planner.resolve_spec(ExecutionContextBuilder().with_device("cpu").build_spec())
    ref = (
        ModelBuilder()
        .with_identifier("fixture/stub-lm")
        .with_family(ModelFamily.QWEN)
        .with_precision(Precision.FP32)
        .build()
    )
    loaded = ModelLoader(ModelBackendFactory().create("stub"), planner).load(
        ref, execution=execution
    )
    spec = (
        AdaptationBuilder()
        .with_adapter_type(AdapterType.LORA)
        .with_rank(8)
        .with_alpha(16)
        .with_target_modules(["q_proj"])
        .build()
    )
    adapted = AdaptationApplier(AdaptationStrategyFactory().create("lora")).apply(
        loaded.handle, spec, execution
    )
    return adapted.handle, execution, loaded.fingerprint.digest, adapted.fingerprint.digest


def build_stub_training_context(
    *,
    config: ExperimentConfig,
    stop_at_step: int | None = None,
    stop_after_steps: int | None = None,
) -> TrainingContext:
    """Assemble a fully bound stub TrainingContext."""
    model, execution, model_fp, adapter_fp = prepare_stub_trainable(seed=config.seed)
    store = CheckpointStoreFactory().create("stub")
    rng = RngControllerFactory().create("python")
    rng.seed_all(config.seed)
    manager = CheckpointManager(checkpoint_store=store, rng=rng)
    bus = TrainingEventBus()
    history = TrainingHistory()
    collector = MetricCollector(history=history)
    trainer_backend = TrainerBackendFactory().create("stub")
    if not isinstance(trainer_backend, StubTrainerBackend):
        raise TypeError("Expected StubTrainerBackend from trainer registry key 'stub'.")
    trainer: StubTrainerBackend = trainer_backend
    if stop_after_steps is not None:
        trainer = StubTrainerBackend(stop_after_steps=stop_after_steps)

    run_id = RunId(value="run-stub")
    experiment_id = config.experiment_id or ExperimentId(value=config.name)
    dataset_session = DatasetSession(
        session_id=f"ds-{uuid4().hex[:8]}",
        experiment_id=experiment_id,
        run_id=run_id,
        shuffle_seed=config.seed,
    )
    training_session = TrainingSession(
        session_id=f"ts-{uuid4().hex[:8]}",
        experiment_id=experiment_id,
        run_id=run_id,
        status=TrainingStatus.PENDING,
        max_steps=config.optimization.max_steps,
        dataset_session=dataset_session,
        execution_digest=f"{execution.selected_device.value}",
        model_fingerprint=model_fp,
        adapter_fingerprint=adapter_fp,
        created_at=datetime.now(UTC),
    )
    bind_extra: dict[str, object] = {}
    if stop_at_step is not None:
        bind_extra["stop_at_step"] = stop_at_step
    if stop_after_steps is not None:
        bind_extra["stop_after_steps"] = stop_after_steps

    from aiodoo_training.domain.training_policies import (
        GradientAccumulationPolicy,
        GradientClippingPolicy,
        LossScalingPolicy,
        MixedPrecisionPolicy,
        OptimizerPolicy,
        SchedulerPolicy,
    )

    ctx = TrainingContext(
        config=config,
        execution=execution,
        model=model,
        dataset_session=dataset_session,
        training_session=training_session,
        trainer=trainer,
        checkpoint_store=store,
        rng=rng,
        optimizer_policy=OptimizerPolicy(learning_rate=config.optimization.learning_rate),
        scheduler_policy=SchedulerPolicy(warmup_ratio=config.optimization.warmup_ratio),
        gradient_accumulation_policy=GradientAccumulationPolicy(
            steps=config.optimization.gradient_accumulation_steps
        ),
        gradient_clipping_policy=GradientClippingPolicy(max_norm=1.0),
        mixed_precision_policy=MixedPrecisionPolicy(precision=config.precision.precision),
        loss_scaling_policy=LossScalingPolicy(enabled=False),
        checkpoint_policy=CheckpointPolicy(
            save_steps=config.checkpointing.save_steps,
            save_total_limit=config.checkpointing.save_total_limit,
        ),
        event_bus=bus,
        checkpoint_manager=manager,
        metric_collector=collector,
        training_history=history,
        trainer_backend_key="stub",
        model_fingerprint=model_fp,
        adapter_fingerprint=adapter_fp,
        config_fingerprint=f"cfg-{config.name}-{config.seed}",
        execution_digest=f"{execution.selected_device.value}",
        adaptation_strategy_key="lora",
        bind_extra=bind_extra,
    )
    trainer.bind(ctx)
    return ctx


def run_stub_train(ctx: TrainingContext) -> tuple[TrainingContext, TrainingProgress]:
    progress = ctx.trainer.train(ctx.config, ctx.model, ctx.execution)
    updated = getattr(ctx.trainer, "context", None)
    if isinstance(updated, TrainingContext):
        return updated, progress
    return ctx, progress


def resume_from_checkpoint(
    *,
    config: ExperimentConfig,
    checkpoint_path: Path,
    policy: ResumePolicy = ResumePolicy.STRICT,
) -> tuple[TrainingContext, ResumeBundle, TrainingProgress]:
    """Validate checkpoint, restore RNG/model/session, continue training."""
    ctx = build_stub_training_context(config=config)
    assert ctx.checkpoint_manager is not None
    coordinator = ResumeCoordinator(
        checkpoint_store=ctx.checkpoint_store,
        rng=ctx.rng,
        checkpoint_manager=ctx.checkpoint_manager,
    )
    expected = ResumeValidationContext(
        experiment_id=ctx.training_session.experiment_id,
        model_fingerprint=ctx.model_fingerprint,
        adapter_fingerprint=ctx.adapter_fingerprint,
        config_fingerprint=ctx.config_fingerprint,
        execution_digest=ctx.execution_digest,
        trainer_backend_key="stub",
    )
    bundle = coordinator.load_and_validate(
        checkpoint_path,
        expected=expected,
        policy=policy,
        training_session=ctx.training_session,
    )
    coordinator.apply_rng(bundle)
    ctx = (
        ctx.with_model(bundle.model)
        .with_training_session(bundle.training_session.with_status(TrainingStatus.PAUSED))
        .with_dataset_session(bundle.dataset_session)
    )
    # Clear interrupt hooks for resume continuation.
    ctx = replace(ctx, bind_extra={})
    if isinstance(ctx.trainer, StubTrainerBackend):
        ctx.trainer.bind(ctx)
    progress = ctx.trainer.resume(ctx.config, ctx.model, bundle.checkpoint, ctx.execution)
    updated = getattr(ctx.trainer, "context", None)
    if isinstance(updated, TrainingContext):
        ctx = updated
    return ctx, bundle, progress
