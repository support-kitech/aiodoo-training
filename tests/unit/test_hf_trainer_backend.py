"""Unit tests for HFTrainerBackend (mocked transformers — no GPU)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aiodoo_training.domain.enums import DeviceKind, Precision, TrainingStatus
from aiodoo_training.domain.examples import IGNORE_INDEX, TokenBatch
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.domain.training import CheckpointHandle, MetricSnapshot, TrainingProgress
from aiodoo_training.domain.training_policies import (
    CheckpointPolicy,
    GradientAccumulationPolicy,
    GradientClippingPolicy,
    LossScalingPolicy,
    MixedPrecisionPolicy,
    OptimizerPolicy,
    SchedulerPolicy,
)
from aiodoo_training.domain.training_session import TrainingSession
from aiodoo_training.exceptions import DomainError
from aiodoo_training.infrastructure.huggingface.trainer import (
    HFTrainerBackend,
    _normalize_token_batches,
    _token_batches_to_dataset,
)
from aiodoo_training.infrastructure.model_handles import (
    OpaqueTrainableModel,
    as_trainable_handle,
)
from aiodoo_training.training.context import TrainingContext
from aiodoo_training.training.engine import make_stub_experiment_config


def _sample_batch() -> TokenBatch:
    return TokenBatch(
        example_ids=("ex-1",),
        input_ids=((11, 22, 33, 0),),
        attention_mask=((1, 1, 1, 0),),
        labels=((11, 22, IGNORE_INDEX, IGNORE_INDEX),),
    )


def _mock_schedule_plan() -> SimpleNamespace:
    return SimpleNamespace(ordered_examples=(), token_batches=(_sample_batch(),))


def _execution() -> ExecutionEnvironment:
    from aiodoo_training.domain.resources import (
        DevicePolicy,
        HardwareCapabilities,
        MemoryPolicy,
        PrecisionPolicy,
    )

    return ExecutionEnvironment(
        selected_device=DeviceKind.CPU,
        device_policy=DevicePolicy(),
        precision_policy=PrecisionPolicy(compute=Precision.FP32),
        memory_policy=MemoryPolicy(),
        capabilities=HardwareCapabilities(),
    )


def _training_context(tmp_path: Path, *, bind_extra: dict | None = None) -> TrainingContext:
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=2, save_steps=1)
    cfg = replace(
        cfg,
        optimization=replace(cfg.optimization, max_steps=2, per_device_batch_size=1),
        checkpointing=replace(cfg.checkpointing, save_steps=1),
    )
    experiment_id = cfg.experiment_id or ExperimentId(value=cfg.name)
    run_id = RunId(value="hf-test")
    dataset_session = DatasetSession(
        session_id="ds-hf",
        experiment_id=experiment_id,
        run_id=run_id,
    )
    training_session = TrainingSession(
        session_id="ts-hf",
        experiment_id=experiment_id,
        run_id=run_id,
        status=TrainingStatus.PENDING,
        max_steps=2,
        dataset_session=dataset_session,
        created_at=datetime.now(UTC),
    )
    extra = {
        "token_batches": (_sample_batch(),),
        "tokenizer": SimpleNamespace(_tokenizer=SimpleNamespace(vocab_size=1000)),
        "schedule_plan": _mock_schedule_plan(),
    }
    if bind_extra:
        extra.update(bind_extra)

    carrier = OpaqueTrainableModel(
        framework_model=MagicMock(),
        aiodoo_adapter_metadata=SimpleNamespace(),
        base=None,
        strategy_key="lora",
    )
    return TrainingContext(
        config=cfg,
        execution=_execution(),
        model=as_trainable_handle(carrier),
        dataset_session=dataset_session,
        training_session=training_session,
        trainer=HFTrainerBackend(),
        checkpoint_store=MagicMock(),
        rng=MagicMock(),
        optimizer_policy=OptimizerPolicy(learning_rate=1e-4),
        scheduler_policy=SchedulerPolicy(warmup_ratio=0.0),
        gradient_accumulation_policy=GradientAccumulationPolicy(steps=1),
        gradient_clipping_policy=GradientClippingPolicy(max_norm=1.0),
        mixed_precision_policy=MixedPrecisionPolicy(precision=Precision.FP32),
        loss_scaling_policy=LossScalingPolicy(enabled=False),
        checkpoint_policy=CheckpointPolicy(save_steps=1, save_total_limit=2),
        trainer_backend_key="hf_trainer",
        bind_extra=extra,
    )


def _mock_transformers_module() -> SimpleNamespace:
    state = SimpleNamespace(global_step=2, epoch=1.0)

    class _FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.state = state
            self._callbacks: list[object] = []

        def add_callback(self, callback: object) -> None:
            self._callbacks.append(callback)

        def train(self, *, resume_from_checkpoint: str | None = None) -> None:
            _ = resume_from_checkpoint
            for callback in self._callbacks:
                if hasattr(callback, "on_train_begin"):
                    callback.on_train_begin(
                        SimpleNamespace(resume_from_checkpoint=resume_from_checkpoint),
                        state,
                        None,
                    )
                if hasattr(callback, "on_log"):
                    callback.on_log(
                        None,
                        state,
                        None,
                        logs={"loss": 0.5},
                    )
                if hasattr(callback, "on_step_end"):
                    callback.on_step_end(None, state, None)
                if hasattr(callback, "on_train_end"):
                    callback.on_train_end(None, state, None)

    return SimpleNamespace(
        Trainer=_FakeTrainer,
        TrainingArguments=lambda **kwargs: SimpleNamespace(**kwargs),
        TrainerCallback=type("TrainerCallback", (), {}),
        default_data_collator=MagicMock(),
    )


def test_normalize_token_batches_accepts_single_and_tuple() -> None:
    batch = _sample_batch()
    assert _normalize_token_batches(batch) == (batch,)
    assert _normalize_token_batches((batch,)) == (batch,)


def test_token_batches_to_dataset_preserves_ids() -> None:
    datasets = MagicMock()
    datasets.Dataset.from_list = MagicMock(side_effect=lambda rows: rows)
    batch = _sample_batch()
    with patch(
        "aiodoo_training.infrastructure.huggingface.trainer._require_datasets",
        return_value=datasets,
    ):
        rows = _token_batches_to_dataset((batch,))
    datasets.Dataset.from_list.assert_called_once()
    payload = datasets.Dataset.from_list.call_args[0][0]
    assert payload[0]["input_ids"] == [11, 22, 33, 0]
    assert payload[0]["labels"] == [11, 22, IGNORE_INDEX, IGNORE_INDEX]


def test_train_requires_bind_extra_fields(tmp_path: Path) -> None:
    ctx = _training_context(tmp_path, bind_extra={"token_batches": None})
    backend = HFTrainerBackend(context=ctx)
    with pytest.raises(DomainError, match="token_batches"):
        backend.train(ctx.config, ctx.model, ctx.execution)


def test_train_runs_transformers_trainer(tmp_path: Path) -> None:
    ctx = _training_context(tmp_path)
    manager = MagicMock()
    manager.save.return_value = CheckpointHandle(
        path=tmp_path / "ckpt-1",
        experiment_id=ctx.training_session.experiment_id,
        run_id=ctx.training_session.run_id,
        checkpoint_type=__import__(
            "aiodoo_training.domain.enums", fromlist=["CheckpointType"]
        ).CheckpointType.FULL_STATE,
        global_step=2,
    )
    ctx = replace(ctx, checkpoint_manager=manager)
    backend = HFTrainerBackend(context=ctx)

    fake_transformers = _mock_transformers_module()
    with patch(
        "aiodoo_training.infrastructure.huggingface.trainer._require_transformers",
        return_value=fake_transformers,
    ):
        progress = backend.train(ctx.config, ctx.model, ctx.execution)

    assert progress.status is TrainingStatus.COMPLETED
    assert progress.global_step == 2
    assert progress.metrics
    assert progress.metrics[0].name == "loss"
    manager.save.assert_called()


def test_resume_passes_checkpoint_path(tmp_path: Path) -> None:
    ctx = _training_context(tmp_path)
    backend = HFTrainerBackend(context=ctx)
    captured: dict[str, str | None] = {}

    class _CapturingTrainer(_mock_transformers_module().Trainer):
        def train(self, *, resume_from_checkpoint: str | None = None) -> None:
            captured["resume"] = resume_from_checkpoint

    fake_transformers = _mock_transformers_module()
    fake_transformers.Trainer = _CapturingTrainer

    checkpoint = CheckpointHandle(
        path=tmp_path / "resume-ckpt",
        experiment_id=ctx.training_session.experiment_id,
        run_id=ctx.training_session.run_id,
        checkpoint_type=__import__(
            "aiodoo_training.domain.enums", fromlist=["CheckpointType"]
        ).CheckpointType.FULL_STATE,
        global_step=5,
    )

    with patch(
        "aiodoo_training.infrastructure.huggingface.trainer._require_transformers",
        return_value=fake_transformers,
    ):
        progress = backend.resume(ctx.config, ctx.model, checkpoint, ctx.execution)

    assert captured["resume"] == str(checkpoint.path)
    assert progress.status is TrainingStatus.COMPLETED


def test_rejects_stub_framework_model(tmp_path: Path) -> None:
    ctx = _training_context(tmp_path)
    stub_carrier = OpaqueTrainableModel(
        framework_model={"kind": "stub", "weights": [1.0]},
        aiodoo_adapter_metadata=SimpleNamespace(),
        base=None,
        strategy_key="lora",
    )
    backend = HFTrainerBackend(context=ctx)
    with pytest.raises(DomainError, match="stub framework"):
        backend.train(ctx.config, as_trainable_handle(stub_carrier), ctx.execution)


def test_checkpoint_save_propagates_persistence_errors(tmp_path: Path) -> None:
    ctx = _training_context(tmp_path)
    manager = MagicMock()
    manager.save.side_effect = OSError("disk full")
    ctx = replace(ctx, checkpoint_manager=manager)
    backend = HFTrainerBackend(context=ctx)
    progress = TrainingProgress(
        status=TrainingStatus.RUNNING,
        global_step=1,
        epoch=0.0,
    )
    with pytest.raises(OSError, match="disk full"):
        backend._request_checkpoint(
            model=ctx.model,
            progress=progress,
            session=ctx.training_session,
            metrics=(),
        )
