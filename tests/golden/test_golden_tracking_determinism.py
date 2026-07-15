"""Golden / determinism: tracking on vs off leaves authoritative surfaces unchanged."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase7
from aiodoo_training.domain.enums import StageStatus, TrainingStatus
from aiodoo_training.domain.identifiers import RunId
from aiodoo_training.pipeline import Pipeline, PipelineContext, build_phase4_pipeline
from aiodoo_training.training.engine import (
    build_stub_training_context,
    make_stub_experiment_config,
)


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    # Phase 4 pipeline RESOLVE_EXECUTION opens DistributedRuntime (placement registry).
    bootstrap_phase7(overwrite=True)


def _run_pipeline(tmp_path: Path, *, tracking_enabled: bool) -> tuple[object, object]:
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=2, save_steps=100)
    ctx0 = build_stub_training_context(config=cfg)
    raw = {
        "training": {"backend": "stub", "max_steps": 2},
        "resume": {"policy": "strict"},
        "checkpointing": {"save_steps": 100, "save_total_limit": 3},
        "tracking": {
            "backend": "null",
            "enabled": tracking_enabled,
            "root_dir": str(tmp_path / "tracking"),
        },
    }
    pipe = Pipeline(build_phase4_pipeline())
    result = pipe.run(
        PipelineContext(
            experiment_id=cfg.experiment_id,
            run_id=RunId(value="gold"),
            config=cfg,
        ).with_values(
            raw_config=raw,
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
    return result, result.stage_results


def test_tracking_on_off_identical_train_status(tmp_path: Path) -> None:
    off, stages_off = _run_pipeline(tmp_path / "off", tracking_enabled=False)
    on, stages_on = _run_pipeline(tmp_path / "on", tracking_enabled=True)
    assert off.status is TrainingStatus.COMPLETED
    assert on.status is TrainingStatus.COMPLETED
    map_off = {r.stage.value: r.status for r in stages_off}
    map_on = {r.stage.value: r.status for r in stages_on}
    assert map_off["train"] is StageStatus.SUCCEEDED
    assert map_on["train"] is StageStatus.SUCCEEDED
    assert map_off == map_on
