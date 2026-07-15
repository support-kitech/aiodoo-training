"""CPU stub TrainerBackend — deterministic loss loop without Torch."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aiodoo_training.domain.enums import TrainingStatus
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.domain.training import CheckpointHandle, MetricSnapshot, TrainingProgress
from aiodoo_training.domain.training_events import TrainingEvent, TrainingEventKind
from aiodoo_training.domain.training_session import TrainingSession
from aiodoo_training.exceptions import DomainError
from aiodoo_training.infrastructure.model_handles import require_trainable_carrier
from aiodoo_training.ports.trainer import TrainerBackend

if TYPE_CHECKING:
    from aiodoo_training.domain.config import ExperimentConfig
    from aiodoo_training.training.checkpoint_manager import SaveCheckpointRequest
    from aiodoo_training.training.context import TrainingContext
else:
    ExperimentConfig = Any  # type: ignore[misc,assignment]
    TrainingContext = Any  # type: ignore[misc,assignment]


DEFAULT_MAX_STEPS = 10


def _ensure_mutable_weights(framework_model: dict[str, Any]) -> list[float]:
    raw = framework_model.get("weights")
    if raw is None:
        weights = [float(i) for i in range(16)]
        framework_model["weights"] = weights
        return weights
    if isinstance(raw, tuple):
        weights = [float(w) for w in raw]
        framework_model["weights"] = weights
        return weights
    if isinstance(raw, list):
        # Mutate in place; coerce to float for stable math.
        for i, value in enumerate(raw):
            raw[i] = float(value)
        return raw
    raise DomainError(f"StubTrainerBackend expects list/tuple weights; got {type(raw).__name__}.")


def stub_loss(*, step: int, weight_sum: float, seed: int) -> float:
    """Deterministic scalar loss used by the CPU stub trainer."""
    return 1.0 / (1.0 + step + weight_sum * 1e-3 + seed * 1e-6)


def _update_weights(weights: list[float], *, loss: float, lr: float) -> None:
    for index in range(len(weights)):
        weights[index] = weights[index] - lr * loss * ((index + 1) * 1e-4)


class StubTrainerBackend(TrainerBackend):
    """
    Deterministic in-process trainer for CPU CI and golden resume tests.

    Rich session collaborators arrive via constructor or :meth:`bind` —
    frozen ``train`` / ``resume`` signatures are unchanged.
    """

    BACKEND_KEY = "stub"

    def __init__(
        self,
        context: TrainingContext | None = None,
        *,
        stop_after_steps: int | None = None,
    ) -> None:
        self._context = context
        self._stop_after_steps = stop_after_steps
        self._lifecycle = None
        try:
            from aiodoo_training.training.lifecycle import TrainingLifecycle

            self._lifecycle = TrainingLifecycle()
        except ImportError:
            self._lifecycle = None

    def bind(self, context: TrainingContext) -> StubTrainerBackend:
        """Attach a resolved :class:`TrainingContext` without widening the port."""
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
        _ = execution
        return self._run_loop(config=config, model=model, start_step=0, resume=False)

    def resume(
        self,
        config: ExperimentConfig,
        model: TrainableModelHandle,
        checkpoint: CheckpointHandle,
        execution: ExecutionEnvironment,
    ) -> TrainingProgress:
        _ = execution
        start_step = max(0, int(checkpoint.global_step))
        return self._run_loop(
            config=config,
            model=model,
            start_step=start_step,
            resume=True,
        )

    def _run_loop(
        self,
        *,
        config: ExperimentConfig,
        model: TrainableModelHandle,
        start_step: int,
        resume: bool,
    ) -> TrainingProgress:
        carrier = require_trainable_carrier(model)
        fw = carrier.framework_model
        if not isinstance(fw, dict):
            raise DomainError(
                "StubTrainerBackend requires OpaqueTrainableModel.framework_model "
                f"to be a dict; got {type(fw).__name__}."
            )
        weights = _ensure_mutable_weights(fw)

        session = self._resolve_session(config, start_step=start_step, resume=resume)
        dataset_session = self._resolve_dataset_session(session)
        seed = int(getattr(config, "seed", 42))
        max_steps = self._resolve_max_steps(config, session)
        stop_after = self._resolve_stop_after()
        stop_at = self._resolve_stop_at_step()
        save_steps = self._resolve_save_steps(config)
        lr = float(getattr(getattr(config, "optimization", None), "learning_rate", 2e-4) or 2e-4)

        metrics: list[MetricSnapshot] = []
        steps_this_call = 0
        status = TrainingStatus.RUNNING
        message: str | None = None

        # Seed / restore RNG for this call when a controller is bound.
        rng = self._context_attr("rng")
        if resume and rng is not None and hasattr(rng, "restore"):
            # Prefer manager-restored state already applied; re-seed only if needed.
            pass
        elif rng is not None and hasattr(rng, "seed_all") and not resume:
            rng.seed_all(seed)

        session = self._transition_start(session, resume=resume)
        self._sync_session(session)
        self._emit(
            TrainingEventKind.TRAINING_STARTED,
            session=session,
            global_step=session.global_step,
            epoch=session.epoch,
        )

        try:
            step = start_step
            while step < max_steps:
                if stop_at is not None and step >= stop_at:
                    status = TrainingStatus.PAUSED
                    message = f"stop_at_step={stop_at}"
                    session = self._transition_pause(session, message=message)
                    break
                if stop_after is not None and steps_this_call >= stop_after:
                    status = TrainingStatus.PAUSED
                    message = f"stop_after_steps={stop_after}"
                    session = self._transition_pause(session, message=message)
                    break

                self._emit(
                    TrainingEventKind.STEP_STARTED,
                    session=session,
                    global_step=step,
                    epoch=session.epoch,
                )

                weight_sum = float(sum(weights))
                loss = stub_loss(step=step, weight_sum=weight_sum, seed=seed)
                _update_weights(weights, loss=loss, lr=lr)

                # Opaque optimizer sidecar for StubCheckpointStore.
                fw["optimizer"] = {
                    "kind": "adamw",
                    "step": step + 1,
                    "learning_rate": lr,
                    "weight_sum": float(sum(weights)),
                }

                now = datetime.now(UTC)
                snapshot = MetricSnapshot(name="loss", value=loss, step=step + 1, timestamp=now)
                metrics.append(snapshot)

                self._emit(
                    TrainingEventKind.LOSS_COMPUTED,
                    session=session,
                    global_step=step + 1,
                    epoch=session.epoch,
                    loss=loss,
                    metrics=(snapshot,),
                )
                self._emit(
                    TrainingEventKind.STEP_COMPLETED,
                    session=session,
                    global_step=step + 1,
                    epoch=session.epoch,
                    loss=loss,
                    metrics=(snapshot,),
                )

                step += 1
                steps_this_call += 1
                epoch = float(step) / float(max_steps) if max_steps else float(step)
                session = session.advance_step(steps=1, epoch=epoch)
                if dataset_session is not None:
                    dataset_session = dataset_session.advance(steps=1)
                self._sync_session(session, dataset_session=dataset_session)

                progress_so_far = TrainingProgress(
                    status=TrainingStatus.RUNNING,
                    global_step=step,
                    epoch=session.epoch,
                    metrics=tuple(metrics),
                )
                if save_steps > 0 and step % save_steps == 0:
                    self._request_checkpoint(
                        model=model,
                        progress=progress_so_far,
                        session=session,
                        dataset_session=dataset_session,
                        metrics=tuple(metrics),
                    )

            if status == TrainingStatus.RUNNING:
                status = TrainingStatus.COMPLETED
                session = self._transition_complete(session)
                message = None
        except Exception as exc:  # noqa: BLE001 — surface as TrainingFailed then re-raise domain
            status = TrainingStatus.FAILED
            message = str(exc)
            session = self._transition_fail(session, message=message)
            fail_progress = TrainingProgress(
                status=status,
                global_step=session.global_step,
                epoch=session.epoch,
                metrics=tuple(metrics),
                message=message,
            )
            self._emit(
                TrainingEventKind.TRAINING_FAILED,
                session=session,
                global_step=session.global_step,
                epoch=session.epoch,
                error=message,
                progress=fail_progress,
            )
            self._sync_session(session)
            raise

        progress = TrainingProgress(
            status=status,
            global_step=session.global_step,
            epoch=session.epoch,
            metrics=tuple(metrics),
            message=message,
        )
        if status == TrainingStatus.COMPLETED:
            self._emit(
                TrainingEventKind.TRAINING_COMPLETED,
                session=session,
                global_step=session.global_step,
                epoch=session.epoch,
                progress=progress,
                metrics=tuple(metrics),
            )
        self._sync_session(session, dataset_session=dataset_session)
        return progress

    # ------------------------------------------------------------------ helpers

    def _context_attr(self, name: str) -> Any:
        ctx = self._context
        if ctx is None:
            return None
        return getattr(ctx, name, None)

    def _resolve_max_steps(self, config: ExperimentConfig, session: TrainingSession) -> int:
        if session.max_steps is not None:
            return int(session.max_steps)
        opt = getattr(config, "optimization", None)
        max_steps = getattr(opt, "max_steps", None) if opt is not None else None
        if max_steps is not None:
            return int(max_steps)
        return DEFAULT_MAX_STEPS

    def _resolve_save_steps(self, config: ExperimentConfig) -> int:
        policy = self._context_attr("checkpoint_policy")
        if policy is not None and getattr(policy, "save_steps", None):
            return int(policy.save_steps)
        ckpt = getattr(config, "checkpointing", None)
        if ckpt is not None and getattr(ckpt, "save_steps", None):
            return int(ckpt.save_steps)
        return 0

    def _resolve_stop_after(self) -> int | None:
        if self._stop_after_steps is not None:
            return int(self._stop_after_steps)
        ctx = self._context
        if ctx is None:
            return None
        extra = getattr(ctx, "bind_extra", None) or {}
        if isinstance(extra, dict) and extra.get("stop_after_steps") is not None:
            return int(extra["stop_after_steps"])
        session = getattr(ctx, "training_session", None)
        if session is not None:
            meta = getattr(session, "metadata", None) or {}
            if "stop_after_steps" in meta:
                return int(meta["stop_after_steps"])
        return None

    def _resolve_stop_at_step(self) -> int | None:
        ctx = self._context
        if ctx is None:
            return None
        extra = getattr(ctx, "bind_extra", None) or {}
        if isinstance(extra, dict) and extra.get("stop_at_step") is not None:
            return int(extra["stop_at_step"])
        session = getattr(ctx, "training_session", None)
        if session is not None:
            meta = getattr(session, "metadata", None) or {}
            if "stop_at_step" in meta:
                return int(meta["stop_at_step"])
        return None

    def _resolve_session(
        self,
        config: ExperimentConfig,
        *,
        start_step: int,
        resume: bool,
    ) -> TrainingSession:
        ctx = self._context
        if ctx is not None and getattr(ctx, "training_session", None) is not None:
            session: TrainingSession = ctx.training_session
            if resume and session.global_step != start_step:
                # Align to checkpoint step when context lags.
                delta = start_step - session.global_step
                if delta > 0:
                    session = session.advance_step(steps=delta)
                elif delta < 0:
                    from dataclasses import replace

                    session = replace(session, global_step=start_step)
            return session

        experiment_id = getattr(config, "experiment_id", None) or ExperimentId(
            value="stub-experiment"
        )
        run_id = RunId(value="stub-run")
        max_steps = getattr(getattr(config, "optimization", None), "max_steps", None)
        return TrainingSession(
            session_id="stub-session",
            experiment_id=experiment_id,
            run_id=run_id,
            status=TrainingStatus.PENDING,
            global_step=start_step,
            epoch=0.0,
            max_steps=max_steps,
        )

    def _resolve_dataset_session(self, session: TrainingSession) -> DatasetSession | None:
        ctx = self._context
        if ctx is not None and getattr(ctx, "dataset_session", None) is not None:
            return ctx.dataset_session
        if session.dataset_session is not None:
            return session.dataset_session
        return DatasetSession(
            session_id=f"{session.session_id}-data",
            experiment_id=session.experiment_id,
            run_id=session.run_id,
        )

    def _sync_session(
        self,
        session: TrainingSession,
        *,
        dataset_session: DatasetSession | None = None,
    ) -> None:
        """Best-effort COW refresh on bound TrainingContext (frozen → replace)."""
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
        # Fresh session constructed locally
        if resume and session.status == TrainingStatus.COMPLETED:
            return session.with_status(TrainingStatus.RUNNING)
        return lifecycle.start(session.with_status(TrainingStatus.PENDING))

    def _transition_pause(
        self, session: TrainingSession, *, message: str | None
    ) -> TrainingSession:
        lifecycle = self._lifecycle
        if lifecycle is None:
            return session.with_status(TrainingStatus.PAUSED, message=message)
        if session.status == TrainingStatus.RUNNING:
            return lifecycle.pause(session, message=message)
        return session.with_status(TrainingStatus.PAUSED, message=message)

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
        except Exception:  # noqa: BLE001
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
        ctx = self._context
        try:
            bus.publish(event, ctx)
        except TypeError:
            bus.publish(event)

    def _request_checkpoint(
        self,
        *,
        model: TrainableModelHandle,
        progress: TrainingProgress,
        session: TrainingSession,
        dataset_session: DatasetSession | None,
        metrics: tuple[MetricSnapshot, ...],
    ) -> CheckpointHandle | None:
        """Ask CheckpointManager to persist — never write files here."""
        manager = self._context_attr("checkpoint_manager")
        if manager is None or not hasattr(manager, "save"):
            return None

        config = self._context_attr("config")
        output_dir = None
        if config is not None:
            ckpt = getattr(config, "checkpointing", None)
            if ckpt is not None:
                output_dir = getattr(ckpt, "output_dir", None)
        if output_dir is None:
            from pathlib import Path

            output_dir = Path("artifacts/checkpoints")

        ds = dataset_session or self._resolve_dataset_session(session)
        ctx = self._context
        request_kwargs = {
            "model": model,
            "progress": progress,
            "training_session": session,
            "dataset_session": ds,
            "experiment_id": session.experiment_id,
            "run_id": session.run_id,
            "output_dir": output_dir,
            "model_fingerprint": getattr(ctx, "model_fingerprint", "") if ctx else "",
            "adapter_fingerprint": getattr(ctx, "adapter_fingerprint", "") if ctx else "",
            "config_fingerprint": getattr(ctx, "config_fingerprint", "") if ctx else "",
            "execution_digest": getattr(ctx, "execution_digest", "") if ctx else "",
            "quantization_digest": getattr(ctx, "quantization_digest", "") if ctx else "",
            "trainer_backend_key": self.BACKEND_KEY,
            "adaptation_strategy_key": getattr(ctx, "adaptation_strategy_key", "") if ctx else "",
            "metrics": metrics,
        }

        try:
            from aiodoo_training.training.checkpoint_manager import SaveCheckpointRequest

            request: SaveCheckpointRequest = SaveCheckpointRequest(**request_kwargs)
            handle = manager.save(request)
        except Exception:  # noqa: BLE001 — duck-type fallback if request shape drifts
            try:
                handle = manager.save(**request_kwargs)
            except TypeError:
                handle = manager.save(model, progress)

        if handle is not None:
            self._emit(
                TrainingEventKind.CHECKPOINT_CREATED,
                session=session,
                global_step=progress.global_step,
                epoch=progress.epoch,
                checkpoint=handle,
                progress=progress,
            )
        return handle


def register_default_trainers(*, overwrite: bool = False) -> None:
    """Register ``stub`` and lazy ``hf_trainer`` backends."""
    from aiodoo_training.infrastructure.huggingface.trainer import register_hf_trainer
    from aiodoo_training.registries import trainer_registry

    if not trainer_registry.exists("stub") or overwrite:
        trainer_registry.register("stub", StubTrainerBackend, overwrite=overwrite)
    register_hf_trainer(overwrite=overwrite)
