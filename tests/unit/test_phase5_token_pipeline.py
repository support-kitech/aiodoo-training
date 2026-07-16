"""Production token pipeline wiring — TokenizeStage → PlanPackingStage → SchedulePlanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase1, bootstrap_phase5
from aiodoo_training.domain.enums import PackingMode, StageStatus
from aiodoo_training.domain.examples import TokenizationConfig
from aiodoo_training.domain.identifiers import RunId
from aiodoo_training.infrastructure.huggingface.tokenizer import DeterministicStubTokenizer
from aiodoo_training.infrastructure.huggingface.templates import QwenChatTemplate
from aiodoo_training.packing.planner import SchedulePlanner
from aiodoo_training.packing.token_rows import (
    TokenRow,
    build_stub_token_row,
    token_batch_to_rows,
)
from aiodoo_training.exceptions import DomainError
from aiodoo_training.pipeline import PipelineContext
from aiodoo_training.pipeline.handlers import CreateTrainerStage, PlanPackingStage, TokenizeStage
from aiodoo_training.training.engine import build_stub_training_context, make_stub_experiment_config
from tests.unit.phase5_helpers import make_examples, plan_once


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase1(overwrite=True)
    bootstrap_phase5(overwrite=True)


def test_tokenize_stage_sets_token_rows() -> None:
    cfg = make_stub_experiment_config(output_dir=Path("/tmp/token-pipe"), max_steps=1, save_steps=100)
    examples = make_examples(2)
    ctx = PipelineContext(
        experiment_id=cfg.experiment_id,
        run_id=RunId(value="tokenize"),
        config=cfg,
    ).with_values(training_examples=examples)

    ctx, result = TokenizeStage().run(ctx)
    assert result.status is StageStatus.SUCCEEDED

    batch = ctx.get("token_batches")
    token_rows = ctx.get("token_rows")
    assert batch is not None
    assert isinstance(token_rows, dict)
    assert token_rows == token_batch_to_rows(batch)


def test_plan_packing_uses_token_rows_from_context() -> None:
    cfg = make_stub_experiment_config(output_dir=Path("/tmp/token-pipe-b"), max_steps=1, save_steps=100)
    examples = make_examples(2)
    config = TokenizationConfig(max_length=128, padding="max_length")
    tokenizer = DeterministicStubTokenizer(template=QwenChatTemplate(), config=config)
    tokenizer.load(cfg.model)
    batch = tokenizer.encode_examples(examples)
    token_rows = token_batch_to_rows(batch)

    ctx = PipelineContext(
        experiment_id=cfg.experiment_id,
        run_id=RunId(value="pack"),
        config=cfg,
    ).with_values(
        training_examples=examples,
        token_batches=batch,
        token_rows=token_rows,
        raw_config={
            "packing": {"backend": "none", "mode": "none", "max_sequence_length": 128},
            "curriculum": {"backend": "none", "mode": "none"},
            "sampling": {"backend": "identity", "seed": 42},
        },
    )

    ctx, result = PlanPackingStage().run(ctx)
    assert result.status is StageStatus.SUCCEEDED

    plan = ctx.get("schedule_plan")
    assert plan is not None
    packed = plan.token_batches[0]
    tokenizer_ids = token_rows[examples[0].example_id].input_ids
    stub_ids = build_stub_token_row(examples[0], max_length=128).input_ids
    assert tokenizer_ids != stub_ids
    assert packed.input_ids[0][: len(tokenizer_ids)] == tokenizer_ids


def test_ensure_order_accepts_provided_token_rows() -> None:
    from aiodoo_training.domain.config import CurriculumSpec, PackingSpec
    from aiodoo_training.domain.enums import CurriculumMode, PackingMode
    from aiodoo_training.domain.identifiers import ExperimentId
    from aiodoo_training.domain.packing_policies import PackingPolicy, SamplingSpec
    from aiodoo_training.factories import (
        CurriculumStrategyFactory,
        PackingStrategyFactory,
        SamplingStrategyFactory,
    )

    examples = make_examples(1)
    provided = {
        examples[0].example_id: TokenRow(
            input_ids=(501, 502, 503),
            attention_mask=(1, 1, 1),
            labels=(501, 502, 503),
        )
    }
    plan = SchedulePlanner().ensure_order(
        examples,
        curriculum=CurriculumStrategyFactory().create("none"),
        sampling=SamplingStrategyFactory().create("identity"),
        packing=PackingStrategyFactory().create("none"),
        curriculum_spec=CurriculumSpec(mode=CurriculumMode.NONE),
        packing_spec=PackingSpec(mode=PackingMode.NONE, max_sequence_length=64),
        sampling_spec=SamplingSpec(backend_key="identity", seed=42),
        packing_policy=PackingPolicy(
            backend_key="none",
            mode=PackingMode.NONE,
            max_sequence_length=64,
        ),
        experiment_id=ExperimentId(value="provided-rows"),
        run_id=RunId(value="run"),
        provided_token_rows=provided,
    )
    row = plan.packing_context.token_rows[examples[0].example_id]
    assert row.input_ids == (501, 502, 503)


def test_stub_path_without_token_rows_unchanged() -> None:
    examples = make_examples(3)
    plan = plan_once(
        examples=examples,
        packing_backend="none",
        packing_mode=PackingMode.NONE,
    )
    stub = build_stub_token_row(examples[0], max_length=64)
    packed = plan.token_batches[0]
    assert packed.input_ids[0][: stub.length] == stub.input_ids


def _plan_packing_context(*, token_rows: dict[str, TokenRow] | None = None) -> PipelineContext:
    cfg = make_stub_experiment_config(output_dir=Path("/tmp/token-guard"), max_steps=1, save_steps=100)
    examples = make_examples(2)
    values: dict[str, object] = {
        "training_examples": examples,
        "raw_config": {
            "training": {"backend": "hf_trainer"},
            "packing": {"backend": "none", "mode": "none", "max_sequence_length": 64},
            "curriculum": {"backend": "none", "mode": "none"},
            "sampling": {"backend": "identity", "seed": 42},
        },
    }
    if token_rows is not None:
        values["token_rows"] = token_rows
    return PipelineContext(
        experiment_id=cfg.experiment_id,
        run_id=RunId(value="guard"),
        config=cfg,
    ).with_values(**values)


def test_plan_packing_hf_trainer_requires_token_rows() -> None:
    with pytest.raises(DomainError, match="token_rows"):
        PlanPackingStage().run(_plan_packing_context())


def test_plan_packing_hf_trainer_accepts_token_rows() -> None:
    examples = make_examples(2)
    token_rows = {
        ex.example_id: TokenRow(
            input_ids=(101, 102),
            attention_mask=(1, 1),
            labels=(101, 102),
        )
        for ex in examples
    }
    ctx, result = PlanPackingStage().run(_plan_packing_context(token_rows=token_rows))
    assert result.status is StageStatus.SUCCEEDED
    assert ctx.get("schedule_plan") is not None


def test_plan_packing_stub_backend_without_token_rows_unchanged() -> None:
    cfg = make_stub_experiment_config(output_dir=Path("/tmp/token-guard-stub"), max_steps=1, save_steps=100)
    examples = make_examples(2)
    ctx = PipelineContext(
        experiment_id=cfg.experiment_id,
        run_id=RunId(value="guard-stub"),
        config=cfg,
    ).with_values(
        training_examples=examples,
        raw_config={
            "training": {"backend": "stub"},
            "packing": {"backend": "none", "mode": "none", "max_sequence_length": 64},
            "curriculum": {"backend": "none", "mode": "none"},
            "sampling": {"backend": "identity", "seed": 42},
        },
    )
    ctx, result = PlanPackingStage().run(ctx)
    assert result.status is StageStatus.SUCCEEDED
    stub = build_stub_token_row(examples[0], max_length=64)
    packed = ctx.get("schedule_plan").token_batches[0]
    assert packed.input_ids[0][: stub.length] == stub.input_ids


def _create_trainer_context(**extra: object) -> PipelineContext:
    cfg = make_stub_experiment_config(output_dir=Path("/tmp/bind-extra"), max_steps=1, save_steps=100)
    stub_ctx = build_stub_training_context(config=cfg)
    values: dict[str, object] = {
        "trainable_model": stub_ctx.model,
        "execution": stub_ctx.execution,
        "dataset_session": stub_ctx.dataset_session,
        "model_fingerprint": stub_ctx.model_fingerprint,
        "adapter_fingerprint": stub_ctx.adapter_fingerprint,
        "config_fingerprint": stub_ctx.config_fingerprint,
        "execution_digest": stub_ctx.execution_digest,
        "adaptation_strategy_key": "lora",
        "raw_config": {"training": {"backend": "hf_trainer"}},
    }
    values.update(extra)
    return PipelineContext(
        experiment_id=cfg.experiment_id,
        run_id=RunId(value="bind-extra"),
        config=cfg,
    ).with_values(**values)


def test_create_trainer_stage_binds_production_objects() -> None:
    examples = make_examples(2)
    config = TokenizationConfig(max_length=64, padding="max_length")
    tokenizer = DeterministicStubTokenizer(template=QwenChatTemplate(), config=config)
    tokenizer.load(make_stub_experiment_config(output_dir=Path("/tmp/x"), max_steps=1, save_steps=1).model)
    batch = tokenizer.encode_examples(examples)
    schedule_plan = plan_once(examples=examples, packing_backend="none", packing_mode=PackingMode.NONE)

    ctx, result = CreateTrainerStage().run(
        _create_trainer_context(
            token_batches=schedule_plan.token_batches,
            tokenizer=tokenizer,
            schedule_plan=schedule_plan,
        )
    )
    assert result.status is StageStatus.SUCCEEDED

    training_context = ctx.get("training_context")
    assert training_context is not None
    assert training_context.bind_extra["token_batches"] is schedule_plan.token_batches
    assert training_context.bind_extra["tokenizer"] is tokenizer
    assert training_context.bind_extra["schedule_plan"] is schedule_plan


def test_create_trainer_stage_preserves_existing_bind_extra() -> None:
    ctx, result = CreateTrainerStage().run(
        _create_trainer_context(bind_extra={"stop_at_step": 3})
    )
    assert result.status is StageStatus.SUCCEEDED
    training_context = ctx.get("training_context")
    assert training_context is not None
    assert training_context.bind_extra["stop_at_step"] == 3
    assert "token_batches" not in training_context.bind_extra


def test_create_trainer_stage_exposes_bind_extra_to_hf_trainer() -> None:
    from aiodoo_training.infrastructure.huggingface.trainer import HFTrainerBackend

    examples = make_examples(1)
    schedule_plan = plan_once(examples=examples, packing_backend="none", packing_mode=PackingMode.NONE)
    ctx, _ = CreateTrainerStage().run(
        _create_trainer_context(
            token_batches=schedule_plan.token_batches,
            schedule_plan=schedule_plan,
        )
    )
    trainer = ctx.get("trainer")
    assert isinstance(trainer, HFTrainerBackend)
    assert trainer.context is not None
    assert trainer.context.bind_extra["schedule_plan"] is schedule_plan
    assert trainer.context.bind_extra["token_batches"] is schedule_plan.token_batches
