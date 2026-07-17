"""Phase 3 pipeline stage handlers — register onto frozen Pipeline only."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from aiodoo_training.config.training_config import (
    GradientFragment,
    to_gradient_policies,
    validate_phase3_fragments,
)
from aiodoo_training.domain.enums import PipelineStage, StageStatus, TrainingStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId, StageName
from aiodoo_training.domain.pipeline import StageResult
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.domain.training_policies import ResumePolicy
from aiodoo_training.domain.training_session import TrainingSession
from aiodoo_training.exceptions import ConfigError, PipelineError
from aiodoo_training.factories.factories import (
    CheckpointStoreFactory,
    ResourcePlannerFactory,
    RngControllerFactory,
    TrainerBackendFactory,
)
from aiodoo_training.pipeline.pipeline import PipelineContext, PipelineStageHandler, require_config
from aiodoo_training.training.checkpoint_manager import (
    CheckpointManager,
    ResumeValidationContext,
)
from aiodoo_training.training.context import TrainingContext
from aiodoo_training.training.event_bus import TrainingEventBus
from aiodoo_training.training.metrics import MetricCollector, TrainingHistory
from aiodoo_training.training.resume import ResumeCoordinator


def _ok(name: StageName, stage: PipelineStage, message: str = "ok") -> StageResult:
    return StageResult(name=name, stage=stage, status=StageStatus.SUCCEEDED, message=message)


def _skip(name: StageName, stage: PipelineStage, message: str) -> StageResult:
    return StageResult(name=name, stage=stage, status=StageStatus.SKIPPED, message=message)


def _export_bind_extra(raw: dict) -> dict[str, object]:
    extra: dict[str, object] = {}
    dataset_version = raw.get("dataset_version")
    if isinstance(dataset_version, str) and dataset_version.strip():
        extra["dataset_version"] = dataset_version
    return extra


class ValidateConfigStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="validate_config")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.VALIDATE_CONFIG

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        config = require_config(context)
        raw = dict(config.metadata) if config.metadata else {}
        # Prefer explicit raw bag if pipeline stored composed YAML.
        composed = context.get("raw_config")
        if isinstance(composed, dict):
            raw = {**composed, **raw}
        try:
            fragments = validate_phase3_fragments(raw if raw else {"training": {"backend": "stub"}})
        except ConfigError as exc:
            raise PipelineError(str(exc)) from exc
        return context.with_values(phase3_fragments=fragments), _ok(self._name, self.stage)


class BootstrapDeterminismStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="bootstrap_determinism")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.BOOTSTRAP_DETERMINISM

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        config = require_config(context)
        rng = context.get("rng")
        if rng is None:
            rng = RngControllerFactory().create("python")
        if context.get("resume_bundle") is None:
            rng.seed_all(config.seed)
        return context.with_values(rng=rng), _ok(self._name, self.stage)


class ResolveExecutionStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="resolve_execution")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.RESOLVE_EXECUTION

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        config = require_config(context)
        updates: dict[str, object] = {}
        if context.get("execution") is None:
            from aiodoo_training.pipeline.stage_helpers import resource_planner_key

            planner_key = resource_planner_key(context)
            planner = ResourcePlannerFactory().create(planner_key)
            execution = planner.resolve_spec(config.execution)
            digest = f"{execution.selected_device.value}:{execution.accelerator.value}"
            updates["execution"] = execution
            updates["execution_digest"] = digest
        else:
            execution = context.get("execution")
            digest = context.get("execution_digest") or (
                f"{execution.selected_device.value}:{execution.accelerator.value}"
            )

        # Phase 7: companion distributed context (handlers only; no stage change).
        if context.get("distributed_context") is None:
            from aiodoo_training.config.distributed_config import (
                parse_phase7_distributed_config,
                to_distributed_spec,
                to_runtime_policy,
                validate_phase7_distributed_fragments,
            )
            from aiodoo_training.distributed.runtime import DistributedRuntime

            # Prefer experiment-config distributed + optional phase7 overlay from metadata.
            dist_spec = config.distributed
            frag_data = {
                "enabled": dist_spec.enabled,
                "backend": "fake",
                "world_size": dist_spec.world_size,
                "num_nodes": dist_spec.num_nodes,
                "global_rank": dist_spec.global_rank,
                "local_rank": dist_spec.local_rank,
            }
            overlay = context.get("distributed_fragment")
            if isinstance(overlay, dict):
                frag_data.update(overlay)
            fragment = parse_phase7_distributed_config(frag_data)
            validate_phase7_distributed_fragments(fragment)
            policy = to_runtime_policy(fragment)
            spec = to_distributed_spec(fragment)
            runtime = DistributedRuntime()
            dist_ctx = runtime.open(policy, execution, spec)
            updates["distributed_context"] = dist_ctx
            updates["distributed_runtime"] = runtime
            updates["mesh_digest"] = dist_ctx.session.topology.mesh_digest
            # Apply DatasetSession placement when session already present.
            session = context.get("dataset_session")
            if session is not None:
                from aiodoo_training.distributed.shard_planner import ShardPlanner

                updates["dataset_session"] = ShardPlanner().apply(
                    session, dist_ctx.session.topology
                )

        if updates:
            context = context.with_values(**updates)
        return context, _ok(self._name, self.stage)


class AssembleDatasetsStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="assemble_datasets")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.ASSEMBLE_DATASETS

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        if context.get("dataset_session") is not None:
            return context, _ok(self._name, self.stage, "dataset_session restored")

        from aiodoo_training.factories.factories import DatasetSourceFactory
        from aiodoo_training.pipeline.stage_helpers import (
            missing_dataset_paths,
            trainer_backend_key,
        )

        config = require_config(context)
        experiment_id = (
            context.experiment_id or config.experiment_id or ExperimentId(value=config.name)
        )
        run_id = context.run_id or RunId(value="run")
        backend = trainer_backend_key(context, config)
        updates: dict[str, object] = {}

        if config.datasets.datasets:
            missing = missing_dataset_paths(config)
            if missing:
                paths = ", ".join(str(path) for path in missing)
                if backend == "stub":
                    # Stub trainers do not require dataset files; keep going.
                    session = DatasetSession(
                        session_id=f"ds-{uuid4().hex[:12]}",
                        experiment_id=experiment_id,
                        run_id=run_id,
                        shuffle_seed=config.seed,
                    )
                    updates["dataset_session"] = session
                    message = f"dataset paths missing ({paths}); stub backend continues"
                else:
                    raise PipelineError(
                        f"Dataset path(s) not found for training backend '{backend}': {paths}"
                    )
            else:
                from aiodoo_training.datasets.fingerprinting import fingerprint_dataset_mix

                source = DatasetSourceFactory().create("jsonl")
                examples = tuple(source.load_mix(config.datasets))
                fingerprint = fingerprint_dataset_mix(
                    config.datasets.datasets,
                    shuffle=config.datasets.shuffle,
                    seed=config.datasets.seed,
                )
                session = DatasetSession(
                    session_id=f"ds-{uuid4().hex[:12]}",
                    experiment_id=experiment_id,
                    run_id=run_id,
                    dataset_fingerprint=fingerprint,
                    mix_fingerprint=fingerprint,
                    examples_total=len(examples),
                    shuffle_seed=config.datasets.seed if config.datasets.shuffle else None,
                )
                updates["dataset_session"] = session
                updates["training_examples"] = examples
                message = f"loaded={len(examples)}"
        else:
            session = DatasetSession(
                session_id=f"ds-{uuid4().hex[:12]}",
                experiment_id=experiment_id,
                run_id=run_id,
                shuffle_seed=config.seed,
            )
            updates["dataset_session"] = session
            message = "ok"

        dist_ctx = context.get("distributed_context")
        if dist_ctx is not None:
            from aiodoo_training.distributed.shard_planner import ShardPlanner

            dataset_session = updates["dataset_session"]
            if not isinstance(dataset_session, DatasetSession):
                raise PipelineError("dataset_session missing before shard planning")
            updates["dataset_session"] = ShardPlanner().apply(
                dataset_session, dist_ctx.session.topology
            )

        return context.with_values(**updates), _ok(self._name, self.stage, message)


class TokenizeStage(PipelineStageHandler):
    """Resolve tokenizer via factory; encode examples when present."""

    def __init__(self) -> None:
        self._name = StageName(value="tokenize")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.TOKENIZE

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        if context.get("token_batches") is not None:
            return context, _ok(self._name, self.stage)

        from aiodoo_training.factories.factories import TokenizerFactory
        from aiodoo_training.pipeline.stage_helpers import model_backend_key, tokenizer_registry_key

        config = require_config(context)
        tokenizer = context.get("tokenizer")
        updates: dict[str, object] = {}
        if tokenizer is None:
            backend = model_backend_key(context, config)
            tokenizer = TokenizerFactory().create(tokenizer_registry_key(backend))
            tokenizer.load(config.model)
            updates["tokenizer"] = tokenizer
            if hasattr(tokenizer, "fingerprint"):
                updates["tokenizer_fingerprint"] = tokenizer.fingerprint()

        examples = context.get("training_examples") or context.get("examples") or ()
        if examples and hasattr(tokenizer, "encode_examples"):
            batch = tokenizer.encode_examples(tuple(examples))
            from aiodoo_training.packing.token_rows import token_batch_to_rows

            updates["token_batches"] = batch
            updates["token_rows"] = token_batch_to_rows(batch)
            return context.with_values(**updates), _ok(self._name, self.stage, "encoded examples")

        if updates:
            return context.with_values(**updates), _ok(
                self._name,
                self.stage,
                "tokenizer resolved; no examples to encode",
            )
        return context, _skip(self._name, self.stage, "No token_batches; skip tokenize.")


class LoadModelStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="load_model")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.LOAD_MODEL

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        if context.get("base_model") is not None or context.get("trainable_model") is not None:
            return context, _ok(self._name, self.stage, "model already present")

        from aiodoo_training.factories.factories import ModelBackendFactory, ResourcePlannerFactory
        from aiodoo_training.models import ModelLoader
        from aiodoo_training.pipeline.stage_helpers import model_backend_key, resource_planner_key

        config = require_config(context)
        backend_key = model_backend_key(context, config)
        planner = ResourcePlannerFactory().create(resource_planner_key(context))
        updates: dict[str, object] = {}
        execution = context.get("execution")
        if execution is None:
            execution = planner.resolve_spec(config.execution)
            updates["execution"] = execution
            updates["execution_digest"] = (
                f"{execution.selected_device.value}:{execution.accelerator.value}"
            )

        loaded = ModelLoader(ModelBackendFactory().create(backend_key), planner).load(
            config.model,
            execution=execution,
        )
        updates["base_model"] = loaded.handle
        updates["model_fingerprint"] = loaded.fingerprint.digest
        updates["model_backend_key"] = backend_key
        return context.with_values(**updates), _ok(self._name, self.stage, f"backend={backend_key}")


class ApplyAdaptationStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="apply_adaptation")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.APPLY_ADAPTATION

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        if context.get("trainable_model") is not None:
            return context, _ok(self._name, self.stage)

        from aiodoo_training.adaptation import AdaptationApplier
        from aiodoo_training.config.model_config import strategy_key_for
        from aiodoo_training.factories.factories import AdaptationStrategyFactory
        from aiodoo_training.pipeline.stage_helpers import raw_config

        config = require_config(context)
        base = context.get("base_model")
        execution = context.get("execution")
        if base is None or execution is None:
            return context, _skip(
                self._name,
                self.stage,
                "base_model and execution required; skipped.",
            )

        # Prefer immutable ExperimentConfig.adaptation; allow raw strategy override.
        strategy_key = config.adaptation.adapter_type.value
        adaptation_raw = raw_config(context).get("adaptation")
        if isinstance(adaptation_raw, dict):
            from aiodoo_training.config.model_config import parse_adaptation_config

            strategy_key = strategy_key_for(parse_adaptation_config(adaptation_raw))
        elif isinstance(config.adaptation.extra.get("strategy"), str):
            strategy_key = str(config.adaptation.extra["strategy"])

        adapted = AdaptationApplier(AdaptationStrategyFactory().create(strategy_key)).apply(
            base, config.adaptation, execution
        )
        return context.with_values(
            trainable_model=adapted.handle,
            adapter_fingerprint=adapted.fingerprint.digest,
            adaptation_strategy_key=strategy_key,
        ), _ok(self._name, self.stage, f"strategy={strategy_key}")


class PlanPackingStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="plan_packing")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PLAN_PACKING

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        from aiodoo_training.domain.config import CurriculumSpec, PackingSpec
        from aiodoo_training.domain.enums import CurriculumMode, PackingMode
        from aiodoo_training.domain.packing_policies import PackingPolicy, SamplingSpec
        from aiodoo_training.factories.factories import (
            CurriculumStrategyFactory,
            PackingStrategyFactory,
            SamplingStrategyFactory,
        )
        from aiodoo_training.packing.planner import SchedulePlanner

        config = require_config(context)
        raw = context.get("raw_config") or {}
        packing_raw = raw.get("packing") if isinstance(raw, dict) else None
        curriculum_raw = raw.get("curriculum") if isinstance(raw, dict) else None
        sampling_raw = raw.get("sampling") if isinstance(raw, dict) else None

        # Existing plan is idempotent short-circuit.
        existing = context.get("schedule_plan")
        if existing is not None:
            return context, _ok(self._name, self.stage, "schedule_plan reused")

        examples = context.get("training_examples") or context.get("examples") or ()
        if not examples:
            # No examples yet — skip packing (tokenization may not have materialised).
            return context, _skip(self._name, self.stage, "no training examples for packing")

        from aiodoo_training.exceptions import DomainError
        from aiodoo_training.pipeline.stage_helpers import trainer_backend_key

        if trainer_backend_key(context, config) == "hf_trainer" and not context.get("token_rows"):
            raise DomainError(
                "PlanPackingStage requires PipelineContext['token_rows'] when "
                "training.backend is hf_trainer."
            )

        packing_backend = "none"
        packing_mode = PackingMode.NONE
        max_seq = config.packing.max_sequence_length
        if isinstance(packing_raw, dict):
            packing_backend = str(packing_raw.get("backend", packing_backend))
            packing_mode = PackingMode(str(packing_raw.get("mode", packing_mode.value)))
            max_seq = int(packing_raw.get("max_sequence_length", max_seq))

        curriculum_backend = "none"
        curriculum_mode = CurriculumMode.NONE
        stages: tuple[str, ...] = ()
        if isinstance(curriculum_raw, dict):
            curriculum_backend = str(curriculum_raw.get("backend", curriculum_backend))
            curriculum_mode = CurriculumMode(str(curriculum_raw.get("mode", curriculum_mode.value)))
            stages = tuple(str(s) for s in (curriculum_raw.get("stages") or ()))

        sampling_backend = "identity"
        sampling_seed = config.seed
        temperature = 1.0
        if isinstance(sampling_raw, dict):
            sampling_backend = str(sampling_raw.get("backend", sampling_backend))
            if sampling_raw.get("seed") is not None:
                sampling_seed = int(sampling_raw["seed"])
            temperature = float(sampling_raw.get("temperature", 1.0))

        packing_spec = PackingSpec(mode=packing_mode, max_sequence_length=max_seq)
        curriculum_spec = CurriculumSpec(mode=curriculum_mode, stages=stages)
        sampling_spec = SamplingSpec(
            backend_key=sampling_backend,
            seed=sampling_seed,
            temperature=temperature,
        )
        packing_policy = PackingPolicy(
            backend_key=packing_backend,
            mode=packing_mode,
            max_sequence_length=max_seq,
            seed=config.seed,
        )

        plan = SchedulePlanner().ensure_order(
            tuple(examples),
            curriculum=CurriculumStrategyFactory().create(curriculum_backend),
            sampling=SamplingStrategyFactory().create(sampling_backend),
            packing=PackingStrategyFactory().create(packing_backend),
            curriculum_spec=curriculum_spec,
            packing_spec=packing_spec,
            sampling_spec=sampling_spec,
            packing_policy=packing_policy,
            experiment_id=(
                context.experiment_id or config.experiment_id or ExperimentId(value=config.name)
            ),
            run_id=context.run_id or RunId(value="phase5-run"),
            seed=config.seed,
            provided_token_rows=context.get("token_rows"),
        )
        return context.with_values(
            schedule_plan=plan,
            packing_session=plan.packing_session,
            curriculum_session=plan.curriculum_session,
            packing_statistics=plan.packing_statistics,
            curriculum_statistics=plan.curriculum_statistics,
            token_batches=plan.token_batches,
            ordered_examples=plan.ordered_examples,
        ), _ok(
            self._name,
            self.stage,
            f"packed={plan.packing_statistics.sequences_emitted} "
            f"stages={plan.curriculum_statistics.stage_count}",
        )


class PlanCurriculumStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="plan_curriculum")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PLAN_CURRICULUM

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        # Idempotent validation — SchedulePlanner already ran in PLAN_PACKING.
        plan = context.get("schedule_plan")
        if plan is None:
            # Delegate to packing stage logic by constructing via planner path.
            packing_stage = PlanPackingStage()
            context, result = packing_stage.run(context)
            if result.status is StageStatus.SKIPPED:
                return context, _skip(
                    self._name, self.stage, result.message or "curriculum skipped"
                )
            plan = context.get("schedule_plan")
        if plan is None:
            return context, _skip(self._name, self.stage, "no schedule_plan")
        return context, _ok(
            self._name,
            self.stage,
            f"curriculum_fp={plan.curriculum_fingerprint[:12]}",
        )


class CreateTrainerStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="create_trainer")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.CREATE_TRAINER

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        config = require_config(context)
        from aiodoo_training.pipeline.stage_helpers import trainer_backend_key

        fragments = context.get("phase3_fragments") or {}
        backend_key = trainer_backend_key(context, config)
        trainer = context.get("trainer") or TrainerBackendFactory().create(backend_key)
        store_key = "hf" if backend_key in {"hf_trainer", "huggingface", "hf"} else "stub"
        store = context.get("checkpoint_store") or CheckpointStoreFactory().create(store_key)
        rng = context.get("rng") or RngControllerFactory().create("python")
        manager = context.get("checkpoint_manager") or CheckpointManager(
            checkpoint_store=store, rng=rng
        )
        bus = context.get("event_bus") or TrainingEventBus()
        history = context.get("training_history") or TrainingHistory()
        collector = context.get("metric_collector") or MetricCollector(history=history)

        experiment_id = (
            context.experiment_id or config.experiment_id or ExperimentId(value=config.name)
        )
        run_id = context.run_id or RunId(value="run")
        dataset_session = context.get("dataset_session")
        if dataset_session is None:
            raise PipelineError("dataset_session required before CREATE_TRAINER")
        trainable = context.get("trainable_model")
        if trainable is None:
            raise PipelineError("trainable_model required before CREATE_TRAINER")
        execution = context.get("execution")
        if execution is None:
            raise PipelineError("execution required before CREATE_TRAINER")

        training_session = context.get("training_session")
        if training_session is None:
            max_steps = config.optimization.max_steps
            training_raw = fragments.get("training")
            if training_raw is not None and getattr(training_raw, "max_steps", None):
                max_steps = training_raw.max_steps
            training_session = TrainingSession(
                session_id=f"train-{uuid4().hex[:12]}",
                experiment_id=experiment_id,
                run_id=run_id,
                status=TrainingStatus.PENDING,
                max_steps=max_steps,
                dataset_session=dataset_session,
                execution_digest=str(context.get("execution_digest") or ""),
                model_fingerprint=str(context.get("model_fingerprint") or ""),
                adapter_fingerprint=str(context.get("adapter_fingerprint") or ""),
                created_at=datetime.now(UTC),
            )

        optimizer_policy = fragments.get("optimizer_policy")
        scheduler_policy = fragments.get("scheduler_policy")
        checkpoint_policy = fragments.get("checkpoint_policy")
        gradient = fragments.get("gradient")
        if not isinstance(gradient, GradientFragment):
            gradient = GradientFragment()
        accum, clip, mixed, loss_scale = to_gradient_policies(
            gradient,
            accumulation_fallback=config.optimization.gradient_accumulation_steps,
            precision=config.precision.precision,
        )
        if optimizer_policy is None or scheduler_policy is None or checkpoint_policy is None:
            # Minimal defaults when VALIDATE_CONFIG was skipped.
            from aiodoo_training.domain.training_policies import (
                CheckpointPolicy,
                OptimizerPolicy,
                SchedulerPolicy,
            )

            optimizer_policy = optimizer_policy or OptimizerPolicy(
                learning_rate=config.optimization.learning_rate,
                weight_decay=config.optimization.weight_decay,
            )
            scheduler_policy = scheduler_policy or SchedulerPolicy(
                warmup_ratio=config.optimization.warmup_ratio,
            )
            checkpoint_policy = checkpoint_policy or CheckpointPolicy(
                save_steps=config.checkpointing.save_steps,
                save_total_limit=config.checkpointing.save_total_limit,
            )

        bind_extra = dict(context.get("bind_extra") or {})
        for key in ("token_batches", "tokenizer", "schedule_plan"):
            value = context.get(key)
            if value is not None:
                bind_extra[key] = value

        training_context = TrainingContext(
            config=config,
            execution=execution,
            model=trainable,
            dataset_session=dataset_session,
            training_session=training_session,
            trainer=trainer,
            checkpoint_store=store,
            rng=rng,
            optimizer_policy=optimizer_policy,
            scheduler_policy=scheduler_policy,
            gradient_accumulation_policy=accum,
            gradient_clipping_policy=clip,
            mixed_precision_policy=mixed,
            loss_scaling_policy=loss_scale,
            checkpoint_policy=checkpoint_policy,
            event_bus=bus,
            checkpoint_manager=manager,
            metric_collector=collector,
            training_history=history,
            trainer_backend_key=backend_key,
            model_fingerprint=str(context.get("model_fingerprint") or ""),
            adapter_fingerprint=str(context.get("adapter_fingerprint") or ""),
            config_fingerprint=str(context.get("config_fingerprint") or ""),
            execution_digest=str(context.get("execution_digest") or ""),
            quantization_digest=str(context.get("quantization_digest") or ""),
            adaptation_strategy_key=str(context.get("adaptation_strategy_key") or ""),
            bind_extra=bind_extra,
        )
        if hasattr(trainer, "bind"):
            trainer.bind(training_context)

        updated = context.with_values(
            trainer=trainer,
            checkpoint_store=store,
            checkpoint_manager=manager,
            event_bus=bus,
            metric_collector=collector,
            training_history=history,
            training_session=training_session,
            training_context=training_context,
            resume_policy=fragments.get("resume_policy") or ResumePolicy.STRICT,
        )
        from aiodoo_training.pipeline.tracking_hooks import maybe_open_tracking

        updated = maybe_open_tracking(updated)
        return updated, _ok(self._name, self.stage)


class RestoreCheckpointStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="restore_checkpoint")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.RESTORE_CHECKPOINT

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        config = require_config(context)
        resume_from = config.checkpointing.resume_from
        if resume_from is None:
            return context, _skip(self._name, self.stage, "Fresh train — no resume_from.")

        manager: CheckpointManager | None = context.get("checkpoint_manager")
        training_session: TrainingSession | None = context.get("training_session")
        rng = context.get("rng")
        if manager is None or training_session is None or rng is None:
            raise PipelineError(
                "checkpoint_manager, training_session, and rng required for resume."
            )

        policy = context.get("resume_policy") or ResumePolicy.STRICT
        expected = ResumeValidationContext(
            experiment_id=training_session.experiment_id,
            model_fingerprint=str(context.get("model_fingerprint") or ""),
            adapter_fingerprint=str(context.get("adapter_fingerprint") or ""),
            config_fingerprint=str(context.get("config_fingerprint") or ""),
            execution_digest=str(context.get("execution_digest") or ""),
            trainer_backend_key=str(context.get("training_context").trainer_backend_key)
            if context.get("training_context") is not None
            else "stub",
        )
        coordinator = ResumeCoordinator(
            checkpoint_store=context.get("checkpoint_store"),
            rng=rng,
            checkpoint_manager=manager,
        )
        bundle = coordinator.load_and_validate(
            Path(resume_from),
            expected=expected,
            policy=policy,
            training_session=training_session,
        )
        coordinator.apply_rng(bundle)
        training_context = context.get("training_context")
        if training_context is not None:
            training_context = (
                training_context.with_model(bundle.model)
                .with_training_session(bundle.training_session)
                .with_dataset_session(bundle.dataset_session)
            )
            trainer = training_context.trainer
            if hasattr(trainer, "bind"):
                trainer.bind(training_context)

        return context.with_values(
            resume_bundle=bundle,
            trainable_model=bundle.model,
            training_session=bundle.training_session,
            dataset_session=bundle.dataset_session,
            checkpoint_handle=bundle.checkpoint,
            training_context=training_context,
        ), _ok(self._name, self.stage, f"Restored {resume_from}")


class TrainStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="train")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.TRAIN

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        config = require_config(context)
        trainer = context.get("trainer")
        model = context.get("trainable_model")
        execution = context.get("execution")
        if trainer is None or model is None or execution is None:
            raise PipelineError("trainer, trainable_model, and execution required for TRAIN.")

        bundle = context.get("resume_bundle")
        handle = context.get("checkpoint_handle")
        if bundle is not None and handle is not None:
            progress = trainer.resume(config, model, handle, execution)
        else:
            progress = trainer.train(config, model, execution)

        training_context = context.get("training_context")
        updated = context.with_values(
            training_progress=progress,
            training_context=training_context,
            training_session=getattr(training_context, "training_session", None)
            if training_context is not None
            else context.get("training_session"),
        )
        from aiodoo_training.pipeline.tracking_hooks import maybe_observe_progress

        maybe_observe_progress(updated)
        return updated, _ok(
            self._name, self.stage, f"status={progress.status.value} step={progress.global_step}"
        )


class EvaluateStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="evaluate")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.EVALUATE

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        from aiodoo_training.builders.evaluation_builders import (
            EvaluationContextBuilder,
            make_evaluation_session,
        )
        from aiodoo_training.config.evaluation_config import (
            parse_evaluation_config,
            to_acceptance_policy,
            to_evaluation_policy,
        )
        from aiodoo_training.domain.config import EvaluationSpec
        from aiodoo_training.evaluation.engine import EvaluationEngine
        from aiodoo_training.factories import EvaluatorFactory

        config = require_config(context)
        raw = context.get("raw_config") or {}
        eval_raw = raw.get("evaluation") if isinstance(raw, dict) else None
        eval_fragment = parse_evaluation_config(eval_raw if isinstance(eval_raw, dict) else {})

        evaluation_spec = config.evaluation
        if not evaluation_spec.enabled and not eval_fragment.enabled:
            return context, _skip(self._name, self.stage, "evaluation disabled")

        # Enable when fragment requests evaluation even if frozen spec defaults disabled.
        if not evaluation_spec.enabled and eval_fragment.enabled:
            from dataclasses import replace

            evaluation_spec = replace(evaluation_spec, enabled=True)

        model = context.get("trainable_model")
        execution = context.get("execution")
        if model is None or execution is None:
            raise PipelineError("trainable_model and execution required for EVALUATE.")

        dataset_refs = evaluation_spec.dataset_refs or ()
        if not dataset_refs and not evaluation_spec.enabled:
            return context, _skip(self._name, self.stage, "no evaluation datasets")

        backend_key = eval_fragment.backend
        evaluator = context.get("evaluator") or EvaluatorFactory().create(backend_key)
        policy = to_evaluation_policy(eval_fragment)
        if policy.seed is None:
            from dataclasses import replace as _replace

            policy = _replace(policy, seed=config.seed)
        acceptance = to_acceptance_policy(eval_fragment)
        run_id = context.run_id or RunId(value="eval-run")
        experiment_id = context.experiment_id or ExperimentId(value=config.name)
        session = context.get("evaluation_session") or make_evaluation_session(
            experiment_id=experiment_id,
            run_id=run_id,
            model_fingerprint=str(context.get("model_fingerprint") or ""),
            adapter_fingerprint=str(context.get("adapter_fingerprint") or ""),
            config_fingerprint=str(context.get("config_fingerprint") or ""),
            execution_digest=str(context.get("execution_digest") or ""),
        )
        eval_ctx = (
            EvaluationContextBuilder()
            .with_config(config)
            .with_piece("execution", execution)
            .with_piece("model", model)
            .with_piece("evaluator", evaluator)
            .with_piece("evaluation_session", session)
            .with_piece("evaluation_spec", evaluation_spec)
            .with_piece("evaluation_policy", policy)
            .with_piece("acceptance_policy", acceptance)
            .with_piece("dataset_refs", dataset_refs)
            .with_piece("evaluator_backend_key", backend_key)
            .with_piece("model_fingerprint", str(context.get("model_fingerprint") or ""))
            .with_piece("adapter_fingerprint", str(context.get("adapter_fingerprint") or ""))
            .with_piece("config_fingerprint", str(context.get("config_fingerprint") or ""))
            .with_piece("execution_digest", str(context.get("execution_digest") or ""))
            .with_piece("rng", context.get("rng"))
            .with_piece("tracker", context.get("tracker"))
            .build()
        )
        _ = EvaluationSpec  # documented companion type
        updated, report = EvaluationEngine().run(eval_ctx)
        quality = EvaluationEngine().validate_gates(report, updated)
        out = context.with_values(
            evaluation_context=updated,
            evaluation_session=updated.evaluation_session,
            evaluation_report=report,
            quality_report=quality,
            evaluator=evaluator,
        )
        from aiodoo_training.pipeline.tracking_hooks import maybe_observe_evaluation

        maybe_observe_evaluation(out)
        return out, _ok(
            self._name,
            self.stage,
            f"passed={report.passed} metrics={len(report.metrics)} gates={quality.passed}",
        )


class ExportStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="export")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.EXPORT

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        from aiodoo_training.builders.export_builders import (
            ExportContextBuilder,
            make_export_session,
        )
        from aiodoo_training.config.export_config import (
            parse_export_config,
            to_compatibility_policy,
            to_validation_policy,
        )
        from aiodoo_training.export.manager import ExportManager
        from aiodoo_training.factories import ExporterFactory

        config = require_config(context)
        raw = context.get("raw_config") or {}
        export_raw = raw.get("export") if isinstance(raw, dict) else None
        export_fragment = parse_export_config(export_raw if isinstance(export_raw, dict) else None)

        if not export_fragment.enabled and context.get("force_export") is not True:
            return context, _skip(self._name, self.stage, "export not requested")

        model = context.get("trainable_model")
        execution = context.get("execution")
        if model is None or execution is None:
            raise PipelineError("trainable_model and execution required for EXPORT.")

        output_dir = export_fragment.output_dir or config.export.output_dir
        output_dir = Path(output_dir)

        backend_key = export_fragment.backend
        exporter = context.get("exporter") or ExporterFactory().create(backend_key)
        experiment_id = context.experiment_id or ExperimentId(value=config.name)
        run_id = context.run_id or RunId(value="export-run")
        session = context.get("export_session") or make_export_session(
            experiment_id=experiment_id,
            run_id=run_id,
            model_fingerprint=str(context.get("model_fingerprint") or ""),
            adapter_fingerprint=str(context.get("adapter_fingerprint") or ""),
            config_fingerprint=str(context.get("config_fingerprint") or ""),
        )
        export_ctx = (
            ExportContextBuilder()
            .with_config(config)
            .with_piece("execution", execution)
            .with_piece("model", model)
            .with_piece("exporter", exporter)
            .with_piece("export_session", session)
            .with_piece("output_dir", output_dir)
            .with_piece("exporter_backend_key", backend_key)
            .with_piece("model_fingerprint", str(context.get("model_fingerprint") or ""))
            .with_piece("adapter_fingerprint", str(context.get("adapter_fingerprint") or ""))
            .with_piece("config_fingerprint", str(context.get("config_fingerprint") or ""))
            .with_piece("evaluation_report", context.get("evaluation_report"))
            .with_piece("quality_report", context.get("quality_report"))
            .with_piece("export_types", tuple(export_fragment.export_types))
            .with_piece("validation_policy", to_validation_policy(export_fragment))
            .with_piece("compatibility_policy", to_compatibility_policy(export_fragment))
            .with_piece("require_evaluation", export_fragment.require_evaluation)
            .with_piece("require_pass_for_export", export_fragment.require_pass_for_export)
            .with_piece("tracker", context.get("tracker"))
            .with_piece(
                "bind_extra",
                _export_bind_extra(raw if isinstance(raw, dict) else {}),
            )
            .build()
        )
        updated, bundle = ExportManager().export(export_ctx)
        out = context.with_values(
            export_context=updated,
            export_session=updated.export_session,
            artifact_bundle=bundle,
            export_manifest=bundle.manifest,
            exporter=exporter,
        )
        from aiodoo_training.pipeline.tracking_hooks import maybe_observe_export

        maybe_observe_export(out)
        return out, _ok(
            self._name,
            self.stage,
            f"bundle={bundle.root.name} fingerprint={bundle.manifest.export_fingerprint[:12]}",
        )


class FinalizeStage(PipelineStageHandler):
    def __init__(self) -> None:
        self._name = StageName(value="finalize")

    @property
    def name(self) -> StageName:
        return self._name

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.FINALIZE

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        history = context.get("training_history")
        path = context.get("metrics_history_path")
        if history is not None and path is not None and hasattr(history, "flush"):
            history.flush(Path(path))
        from aiodoo_training.pipeline.artifact_hooks import maybe_publish_artifacts
        from aiodoo_training.pipeline.tracking_hooks import maybe_finalize_tracking

        maybe_finalize_tracking(context)
        maybe_publish_artifacts(context)
        tracker = context.get("tracker")
        # Coordinator.close already closes the tracker when present.
        if (
            tracker is not None
            and context.get("tracking_coordinator") is None
            and hasattr(tracker, "close")
        ):
            tracker.close()
        runtime = context.get("distributed_runtime")
        if runtime is not None and hasattr(runtime, "close"):
            runtime.close()
        return context, _ok(self._name, self.stage)


def build_phase3_pipeline() -> list[PipelineStageHandler]:
    """Ordered Phase 3 stage handlers for the frozen Pipeline orchestrator."""
    return [
        ValidateConfigStage(),
        BootstrapDeterminismStage(),
        ResolveExecutionStage(),
        AssembleDatasetsStage(),
        TokenizeStage(),
        LoadModelStage(),
        ApplyAdaptationStage(),
        PlanPackingStage(),
        PlanCurriculumStage(),
        CreateTrainerStage(),
        RestoreCheckpointStage(),
        TrainStage(),
        EvaluateStage(),
        ExportStage(),
        FinalizeStage(),
    ]


def build_phase4_pipeline() -> list[PipelineStageHandler]:
    """Phase 4 pipeline: same stage graph with evaluation/export handlers live."""
    return build_phase3_pipeline()
