"""HuggingFace TrainerBackend — production training via transformers.Trainer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiodoo_training.domain.enums import DeviceKind, Precision, TrainingStatus
from aiodoo_training.domain.examples import TokenBatch
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.domain.training import CheckpointHandle, MetricSnapshot, TrainingProgress
from aiodoo_training.domain.training_events import TrainingEvent, TrainingEventKind
from aiodoo_training.domain.training_session import TrainingSession
from aiodoo_training.exceptions import DomainError, FactoryError, TrainingLifecycleError
from aiodoo_training.infrastructure.model_handles import require_trainable_carrier
from aiodoo_training.ports.trainer import TrainerBackend

if TYPE_CHECKING:
    from aiodoo_training.domain.config import ExperimentConfig
    from aiodoo_training.training.context import TrainingContext
else:
    ExperimentConfig = Any  # type: ignore[misc,assignment]
    TrainingContext = Any  # type: ignore[misc,assignment]

_AiodooTrainerCallbackCls: type[Any] | None = None


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise FactoryError(
            "HFTrainerBackend requires the 'transformers' package. "
            "Install training extras or use TrainerBackend key 'stub' for CPU CI."
        ) from exc
    return transformers


def _require_datasets() -> Any:
    try:
        import datasets
    except ImportError as exc:
        raise FactoryError(
            "HFTrainerBackend requires the 'datasets' package. "
            "Install training extras or use TrainerBackend key 'stub' for CPU CI."
        ) from exc
    return datasets


def _normalize_token_batches(raw: object) -> tuple[TokenBatch, ...]:
    if isinstance(raw, TokenBatch):
        return (raw,)
    if isinstance(raw, tuple):
        batches = tuple(item for item in raw if isinstance(item, TokenBatch))
        if batches:
            return batches
    raise DomainError(
        "HFTrainerBackend requires bind_extra['token_batches'] as TokenBatch or "
        "tuple[TokenBatch, ...]."
    )


def _token_batches_to_dataset(token_batches: tuple[TokenBatch, ...]) -> Any:
    datasets = _require_datasets()
    rows: list[dict[str, list[int]]] = []
    for batch in token_batches:
        for input_ids, attention_mask, labels in zip(
            batch.input_ids,
            batch.attention_mask,
            batch.labels,
            strict=True,
        ):
            rows.append(
                {
                    "input_ids": list(input_ids),
                    "attention_mask": list(attention_mask),
                    "labels": list(labels),
                }
            )
    if not rows:
        raise DomainError("HFTrainerBackend requires non-empty token_batches.")
    return datasets.Dataset.from_list(rows)


def _resolve_hf_tokenizer(tokenizer_port: object) -> Any:
    hf_tok = getattr(tokenizer_port, "_tokenizer", None)
    if hf_tok is not None:
        return hf_tok
    encode = getattr(tokenizer_port, "encode", None)
    if callable(encode):
        return tokenizer_port
    raise DomainError(
        "HFTrainerBackend requires bind_extra['tokenizer'] exposing a HuggingFace "
        "AutoTokenizer via _tokenizer."
    )


def _resolve_bind_extra(bind_extra: dict[str, Any] | None) -> tuple[tuple[TokenBatch, ...], object]:
    extra = bind_extra or {}
    token_batches = _normalize_token_batches(extra.get("token_batches"))
    tokenizer_port = extra.get("tokenizer")
    if tokenizer_port is None:
        raise DomainError("HFTrainerBackend requires bind_extra['tokenizer'].")
    if extra.get("schedule_plan") is None:
        raise DomainError("HFTrainerBackend requires bind_extra['schedule_plan'].")
    return token_batches, tokenizer_port


def _reject_stub_framework(framework: object) -> None:
    if isinstance(framework, dict) and framework.get("kind") == "stub":
        raise DomainError(
            "HFTrainerBackend cannot train stub framework models. "
            "Use trainer backend key 'stub' for CPU CI."
        )


def _build_training_arguments(
    *,
    config: ExperimentConfig,
    context: TrainingContext,
    execution: ExecutionEnvironment,
    output_dir: Path,
    resume_from_checkpoint: str | None,
) -> Any:
    """
    Map ExperimentConfig + TrainingContext policies to ``TrainingArguments``.

    Compatible with transformers>=4.51 (pinned in requirements/train.txt).
    ``resume_from_checkpoint`` is stored on args for script parity; the active
    resume path is ``Trainer.train(resume_from_checkpoint=...)``.
    """
    transformers = _require_transformers()
    TrainingArguments = transformers.TrainingArguments

    opt = config.optimization
    optimizer = context.optimizer_policy
    scheduler = context.scheduler_policy
    accum = context.gradient_accumulation_policy
    clip = context.gradient_clipping_policy
    mixed = context.mixed_precision_policy
    checkpoint = context.checkpoint_policy
    session = context.training_session

    max_steps = session.max_steps if session.max_steps is not None else opt.max_steps
    use_cpu = execution.selected_device == DeviceKind.CPU
    fp16 = mixed.precision == Precision.FP16
    bf16 = mixed.precision == Precision.BF16

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": optimizer.learning_rate,
        "weight_decay": optimizer.weight_decay,
        "warmup_ratio": scheduler.warmup_ratio,
        "num_train_epochs": opt.num_epochs,
        "per_device_train_batch_size": opt.per_device_batch_size,
        "gradient_accumulation_steps": accum.steps,
        "logging_steps": max(1, min(checkpoint.save_steps, 10)),
        "save_strategy": "no",
        "report_to": [],
        "seed": config.seed,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "dataloader_num_workers": 0,
        "use_cpu": use_cpu,
        "fp16": fp16 and not use_cpu,
        "bf16": bf16 and not use_cpu,
        "gradient_checkpointing": config.precision.gradient_checkpointing,
        "resume_from_checkpoint": resume_from_checkpoint,
    }
    if max_steps is not None:
        kwargs["max_steps"] = int(max_steps)
    if clip.max_norm is not None:
        kwargs["max_grad_norm"] = float(clip.max_norm)
    return TrainingArguments(**kwargs)


def _create_hf_trainer(
    *,
    framework_model: Any,
    training_args: Any,
    dataset: Any,
    hf_tokenizer: Any,
) -> Any:
    transformers = _require_transformers()
    return transformers.Trainer(
        model=framework_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=hf_tokenizer,
        data_collator=transformers.default_data_collator,
    )


def _aiodoo_trainer_callback_class() -> type[Any]:
    """Lazy ``TrainerCallback`` subclass (transformers imported on first use)."""
    global _AiodooTrainerCallbackCls
    if _AiodooTrainerCallbackCls is not None:
        return _AiodooTrainerCallbackCls

    TrainerCallback = _require_transformers().TrainerCallback

    class AiodooTrainerCallback(TrainerCallback):
        """Bridge HuggingFace Trainer hooks to EventBus and CheckpointManager."""

        def __init__(self, backend: HFTrainerBackend) -> None:
            super().__init__()
            self._backend = backend
            self._metrics: list[MetricSnapshot] = []

        @property
        def metrics(self) -> tuple[MetricSnapshot, ...]:
            return tuple(self._metrics)

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            backend = self._backend
            ctx = backend._require_context()
            session = backend._transition_start(
                ctx.training_session,
                resume=bool(getattr(args, "resume_from_checkpoint", None)),
            )
            backend._sync_session(session)
            backend._emit(
                TrainingEventKind.TRAINING_STARTED,
                session=session,
                global_step=session.global_step,
                epoch=session.epoch,
            )

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if not logs:
                return
            loss_raw = logs.get("loss")
            if loss_raw is None:
                return
            backend = self._backend
            ctx = backend._require_context()
            session = ctx.training_session
            step = int(state.global_step)
            epoch = float(state.epoch or session.epoch)
            loss = float(loss_raw)
            snapshot = MetricSnapshot(
                name="loss",
                value=loss,
                step=step,
                timestamp=datetime.now(UTC),
            )
            self._metrics.append(snapshot)
            backend._emit(
                TrainingEventKind.LOSS_COMPUTED,
                session=session,
                global_step=step,
                epoch=epoch,
                loss=loss,
                metrics=(snapshot,),
            )
            backend._emit(
                TrainingEventKind.STEP_COMPLETED,
                session=session,
                global_step=step,
                epoch=epoch,
                loss=loss,
                metrics=(snapshot,),
            )

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            backend = self._backend
            ctx = backend._require_context()
            session = ctx.training_session
            step = int(state.global_step)
            epoch = float(state.epoch or session.epoch)
            delta = max(0, step - session.global_step)
            if delta:
                session = session.advance_step(steps=delta, epoch=epoch)
                backend._sync_session(session)
            save_steps = ctx.checkpoint_policy.save_steps
            if save_steps > 0 and step > 0 and step % save_steps == 0:
                progress = TrainingProgress(
                    status=TrainingStatus.RUNNING,
                    global_step=step,
                    epoch=epoch,
                    metrics=tuple(self._metrics),
                )
                backend._request_checkpoint(
                    model=ctx.model,
                    progress=progress,
                    session=session,
                    metrics=tuple(self._metrics),
                )

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            _ = (args, state, control, kwargs)

    _AiodooTrainerCallbackCls = AiodooTrainerCallback
    return _AiodooTrainerCallbackCls


def _checkpoint_output_dir(context: TrainingContext | None) -> Path:
    config = getattr(context, "config", None) if context is not None else None
    if config is not None:
        ckpt = getattr(config, "checkpointing", None)
        if ckpt is not None:
            output_dir = getattr(ckpt, "output_dir", None)
            if output_dir is not None:
                return Path(output_dir)
    return Path("artifacts/checkpoints")


def _resolve_dataset_session(
    context: TrainingContext | None,
    session: TrainingSession,
) -> DatasetSession:
    if context is not None and context.dataset_session is not None:
        return context.dataset_session
    if session.dataset_session is not None:
        return session.dataset_session
    return DatasetSession(
        session_id=f"{session.session_id}-data",
        experiment_id=session.experiment_id,
        run_id=session.run_id,
    )


def _save_checkpoint_with_signature_fallbacks(
    manager: Any,
    *,
    request_kwargs: dict[str, Any],
    model: TrainableModelHandle,
    progress: TrainingProgress,
) -> CheckpointHandle | None:
    """Fallback save paths for alternate ``CheckpointManager.save`` signatures only."""
    try:
        return manager.save(**request_kwargs)
    except TypeError:
        return manager.save(model, progress)


def _save_checkpoint_via_manager(
    manager: Any,
    *,
    request_kwargs: dict[str, Any],
    model: TrainableModelHandle,
    progress: TrainingProgress,
) -> CheckpointHandle | None:
    """Persist via ``CheckpointManager.save``; propagate unexpected persistence failures."""
    from aiodoo_training.training.checkpoint_manager import SaveCheckpointRequest

    try:
        request = SaveCheckpointRequest(**request_kwargs)
    except TypeError:
        return _save_checkpoint_with_signature_fallbacks(
            manager,
            request_kwargs=request_kwargs,
            model=model,
            progress=progress,
        )

    try:
        return manager.save(request)
    except TypeError:
        return _save_checkpoint_with_signature_fallbacks(
            manager,
            request_kwargs=request_kwargs,
            model=model,
            progress=progress,
        )


def _publish_training_event(
    bus: Any,
    event: TrainingEvent,
    context: TrainingContext | None,
) -> None:
    try:
        bus.publish(event, context)
    except TypeError:
        bus.publish(event)


class HFTrainerBackend(TrainerBackend):
    """
    Production HuggingFace Trainer adapter.

    Consumes packed ``TokenBatch`` objects from ``TrainingContext.bind_extra``;
    never re-tokenizes. Checkpoint durability flows through ``CheckpointManager``.
    """

    BACKEND_KEY = "hf_trainer"

    def __init__(self, context: TrainingContext | None = None) -> None:
        self._context = context
        self._lifecycle: Any | None = None
        try:
            from aiodoo_training.training.lifecycle import TrainingLifecycle

            self._lifecycle = TrainingLifecycle()
        except ImportError:
            self._lifecycle = None

    def bind(self, context: TrainingContext) -> HFTrainerBackend:
        self._context = context
        return self

    @property
    def context(self) -> TrainingContext | None:
        return self._context

    def train(
        self,
        config: ExperimentConfig,
        model: TrainableModelHandle,
        execution: ExecutionEnvironment,
    ) -> TrainingProgress:
        return self._run_training(
            config=config,
            model=model,
            execution=execution,
            resume_from_checkpoint=None,
        )

    def resume(
        self,
        config: ExperimentConfig,
        model: TrainableModelHandle,
        checkpoint: CheckpointHandle,
        execution: ExecutionEnvironment,
    ) -> TrainingProgress:
        return self._run_training(
            config=config,
            model=model,
            execution=execution,
            resume_from_checkpoint=str(checkpoint.path),
            start_step=max(0, int(checkpoint.global_step)),
        )

    def _run_training(
        self,
        *,
        config: ExperimentConfig,
        model: TrainableModelHandle,
        execution: ExecutionEnvironment,
        resume_from_checkpoint: str | None,
        start_step: int = 0,
    ) -> TrainingProgress:
        ctx = self._require_context()
        carrier = require_trainable_carrier(model)
        framework = carrier.framework_model
        _reject_stub_framework(framework)

        token_batches, tokenizer_port = _resolve_bind_extra(ctx.bind_extra)
        hf_tokenizer = _resolve_hf_tokenizer(tokenizer_port)
        dataset = _token_batches_to_dataset(token_batches)
        output_dir = config.checkpointing.output_dir

        training_args = _build_training_arguments(
            config=config,
            context=ctx,
            execution=execution,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        hf_trainer = _create_hf_trainer(
            framework_model=framework,
            training_args=training_args,
            dataset=dataset,
            hf_tokenizer=hf_tokenizer,
        )
        callback = _aiodoo_trainer_callback_class()(self)
        hf_trainer.add_callback(callback)

        self._align_session_to_start_step(start_step)
        self._seed_rng_for_fresh_run(config, resume_from_checkpoint)

        metrics: tuple[MetricSnapshot, ...] = ()
        try:
            hf_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            return self._build_success_progress(hf_trainer, callback)
        except Exception as exc:
            self._handle_training_failure(exc, metrics)

    def _align_session_to_start_step(self, start_step: int) -> None:
        if start_step <= 0:
            return
        ctx = self._require_context()
        session = ctx.training_session
        if session.global_step == start_step:
            return
        session = session.advance_step(steps=max(0, start_step - session.global_step))
        self._sync_session(session)

    def _seed_rng_for_fresh_run(
        self,
        config: ExperimentConfig,
        resume_from_checkpoint: str | None,
    ) -> None:
        if resume_from_checkpoint is not None:
            return
        ctx = self._require_context()
        rng = ctx.rng
        if rng is not None and hasattr(rng, "seed_all"):
            rng.seed_all(config.seed)

    def _build_success_progress(self, hf_trainer: Any, callback: Any) -> TrainingProgress:
        metrics = callback.metrics
        session = self._require_context().training_session
        global_step = int(getattr(hf_trainer.state, "global_step", session.global_step))
        epoch = float(getattr(hf_trainer.state, "epoch", session.epoch) or session.epoch)
        status = session.status
        if status not in {TrainingStatus.COMPLETED, TrainingStatus.PAUSED}:
            status = TrainingStatus.COMPLETED
            session = self._transition_complete(session)
            self._sync_session(session)
        progress = TrainingProgress(
            status=status,
            global_step=global_step,
            epoch=epoch,
            metrics=metrics,
        )
        self._emit(
            TrainingEventKind.TRAINING_COMPLETED,
            session=session,
            global_step=global_step,
            epoch=epoch,
            progress=progress,
            metrics=metrics,
        )
        return progress

    def _handle_training_failure(
        self,
        exc: Exception,
        metrics: tuple[MetricSnapshot, ...],
    ) -> None:
        session = self._require_context().training_session
        session = self._transition_fail(session, message=str(exc))
        self._sync_session(session)
        progress = TrainingProgress(
            status=TrainingStatus.FAILED,
            global_step=session.global_step,
            epoch=session.epoch,
            metrics=metrics,
            message=str(exc),
        )
        self._emit(
            TrainingEventKind.TRAINING_FAILED,
            session=session,
            global_step=session.global_step,
            epoch=session.epoch,
            error=str(exc),
            progress=progress,
        )
        raise exc

    def _require_context(self) -> TrainingContext:
        if self._context is None:
            raise DomainError("HFTrainerBackend is not bound to a TrainingContext.")
        return self._context

    def _context_attr(self, name: str) -> Any:
        ctx = self._context
        if ctx is None:
            return None
        return getattr(ctx, name, None)

    def _sync_session(
        self,
        session: TrainingSession,
        *,
        dataset_session: DatasetSession | None = None,
    ) -> None:
        ctx = self._context
        if ctx is None:
            return
        updated = ctx
        if hasattr(ctx, "with_training_session"):
            updated = ctx.with_training_session(session)
        if dataset_session is not None and hasattr(updated, "with_dataset_session"):
            updated = updated.with_dataset_session(dataset_session)
        self._context = updated

    def _transition_start(self, session: TrainingSession, *, resume: bool) -> TrainingSession:
        lifecycle = self._lifecycle
        if lifecycle is None:
            return session.with_status(TrainingStatus.RUNNING)
        if session.status == TrainingStatus.PENDING:
            return lifecycle.start(session)
        if session.status == TrainingStatus.PAUSED:
            return lifecycle.resume_running(session)
        if session.status == TrainingStatus.RUNNING:
            return session
        if resume:
            return lifecycle.start(session.with_status(TrainingStatus.PENDING))
        return lifecycle.start(session.with_status(TrainingStatus.PENDING))

    def _transition_complete(self, session: TrainingSession) -> TrainingSession:
        lifecycle = self._lifecycle
        if lifecycle is None:
            return session.with_status(TrainingStatus.COMPLETED)
        if session.status == TrainingStatus.RUNNING:
            return lifecycle.complete(session)
        return session.with_status(TrainingStatus.COMPLETED)

    def _transition_fail(self, session: TrainingSession, *, message: str | None) -> TrainingSession:
        lifecycle = self._lifecycle
        if lifecycle is None:
            return session.with_status(TrainingStatus.FAILED, message=message)
        try:
            return lifecycle.fail(session, message=message)
        except TrainingLifecycleError:
            return session.with_status(TrainingStatus.FAILED, message=message)

    def _emit(
        self,
        kind: TrainingEventKind,
        *,
        session: TrainingSession,
        global_step: int,
        epoch: float,
        loss: float | None = None,
        metrics: tuple[MetricSnapshot, ...] = (),
        progress: TrainingProgress | None = None,
        error: str | None = None,
        checkpoint: CheckpointHandle | None = None,
    ) -> None:
        bus = self._context_attr("event_bus")
        if bus is None or not hasattr(bus, "publish"):
            return
        event = TrainingEvent(
            kind=kind,
            experiment_id=session.experiment_id,
            run_id=session.run_id,
            session_id=session.session_id,
            timestamp=datetime.now(UTC),
            global_step=global_step,
            epoch=epoch,
            loss=loss,
            metrics=metrics,
            checkpoint=checkpoint,
            progress=progress,
            error=error,
        )
        _publish_training_event(bus, event, self._context)

    def _request_checkpoint(
        self,
        *,
        model: TrainableModelHandle,
        progress: TrainingProgress,
        session: TrainingSession,
        metrics: tuple[MetricSnapshot, ...],
    ) -> CheckpointHandle | None:
        manager = self._context_attr("checkpoint_manager")
        if manager is None or not hasattr(manager, "save"):
            return None

        ctx = self._context
        request_kwargs = {
            "model": model,
            "progress": progress,
            "training_session": session,
            "dataset_session": _resolve_dataset_session(ctx, session),
            "experiment_id": session.experiment_id,
            "run_id": session.run_id,
            "output_dir": _checkpoint_output_dir(ctx),
            "model_fingerprint": getattr(ctx, "model_fingerprint", "") if ctx else "",
            "adapter_fingerprint": getattr(ctx, "adapter_fingerprint", "") if ctx else "",
            "config_fingerprint": getattr(ctx, "config_fingerprint", "") if ctx else "",
            "execution_digest": getattr(ctx, "execution_digest", "") if ctx else "",
            "quantization_digest": getattr(ctx, "quantization_digest", "") if ctx else "",
            "trainer_backend_key": self.BACKEND_KEY,
            "adaptation_strategy_key": getattr(ctx, "adaptation_strategy_key", "") if ctx else "",
            "metrics": metrics,
        }

        handle = _save_checkpoint_via_manager(
            manager,
            request_kwargs=request_kwargs,
            model=model,
            progress=progress,
        )
        if handle is not None:
            self._emit(
                TrainingEventKind.CHECKPOINT_CREATED,
                session=session,
                global_step=progress.global_step,
                epoch=session.epoch,
                checkpoint=handle,
                progress=progress,
                metrics=metrics,
            )
        return handle


def register_hf_trainer(*, overwrite: bool = False) -> None:
    """Register the HF trainer under ``hf_trainer`` (lazy — no import of transformers)."""
    from aiodoo_training.registries import trainer_registry

    if not trainer_registry.exists("hf_trainer") or overwrite:
        trainer_registry.register("hf_trainer", HFTrainerBackend, overwrite=overwrite)
