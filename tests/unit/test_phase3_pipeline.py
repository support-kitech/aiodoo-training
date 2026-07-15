"""Phase 3 pipeline handlers + callback smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase3
from aiodoo_training.domain.enums import StageStatus, TrainingStatus
from aiodoo_training.domain.identifiers import RunId
from aiodoo_training.pipeline import Pipeline, PipelineContext, build_phase3_pipeline
from aiodoo_training.training.engine import (
    build_stub_training_context,
    make_stub_experiment_config,
)


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase3(overwrite=True)


def test_phase3_pipeline_runs_stub_train(tmp_path: Path) -> None:
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=3, save_steps=100)
    # Pre-build collaborators the thin LOAD_MODEL stage would skip
    ctx0 = build_stub_training_context(config=cfg)
    pipe = Pipeline(build_phase3_pipeline())
    result = pipe.run(
        PipelineContext(
            experiment_id=cfg.experiment_id,
            run_id=RunId(value="pipe-run"),
            config=cfg,
        ).with_values(
            raw_config={
                "training": {"backend": "stub", "max_steps": 3},
                "resume": {"policy": "strict"},
                "checkpointing": {"save_steps": 100, "save_total_limit": 3},
            },
            trainable_model=ctx0.model,
            execution=ctx0.execution,
            model_fingerprint=ctx0.model_fingerprint,
            adapter_fingerprint=ctx0.adapter_fingerprint,
            config_fingerprint=ctx0.config_fingerprint,
            execution_digest=ctx0.execution_digest,
            adaptation_strategy_key="lora",
            rng=ctx0.rng,
            checkpoint_store=ctx0.checkpoint_store,
            trainer=ctx0.trainer,
        )
    )
    assert result.status is TrainingStatus.COMPLETED
    statuses = {r.stage.value: r.status for r in result.stage_results}
    assert statuses["train"] is StageStatus.SUCCEEDED
    assert statuses["evaluate"] is StageStatus.SKIPPED


def test_logging_callback_registers() -> None:
    from aiodoo_training.factories import CallbackFactory

    cb = CallbackFactory().create("logging")
    assert cb is not None
