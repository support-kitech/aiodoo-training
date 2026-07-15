"""Phase 4 pipeline integration — evaluate/export stages with Phase 3 train path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase7
from aiodoo_training.builders.evaluation_builders import enable_evaluation
from aiodoo_training.domain.enums import DatasetType, StageStatus, TrainingStatus
from aiodoo_training.domain.identifiers import RunId
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.pipeline import Pipeline, PipelineContext, build_phase4_pipeline
from aiodoo_training.training.engine import (
    build_stub_training_context,
    make_stub_experiment_config,
)


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    # build_phase4_pipeline RESOLVE_EXECUTION requires placement registry (Phase 7).
    bootstrap_phase7(overwrite=True)


def _eval_ref(path: Path) -> DatasetRef:
    return DatasetRef(
        path=path,
        dataset_type=DatasetType.EVALUATION,
        protocol_version="1.0",
        name="pipe-eval",
    )


def _base_pipeline_context(tmp_path: Path, *, raw_config: dict | None = None) -> PipelineContext:
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=3, save_steps=100)
    ctx0 = build_stub_training_context(config=cfg)
    raw = raw_config or {
        "training": {"backend": "stub", "max_steps": 3},
        "resume": {"policy": "strict"},
        "checkpointing": {"save_steps": 100, "save_total_limit": 3},
    }
    return PipelineContext(
        experiment_id=cfg.experiment_id,
        run_id=RunId(value="pipe-run"),
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


def test_phase4_pipeline_skips_evaluate_and_export_by_default(tmp_path: Path) -> None:
    pipe = Pipeline(build_phase4_pipeline())
    result = pipe.run(_base_pipeline_context(tmp_path))
    assert result.status is TrainingStatus.COMPLETED
    statuses = {r.stage.value: r.status for r in result.stage_results}
    assert statuses["train"] is StageStatus.SUCCEEDED
    assert statuses["evaluate"] is StageStatus.SKIPPED
    assert statuses["export"] is StageStatus.SKIPPED


def test_phase4_pipeline_runs_evaluate_and_export_when_enabled(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    eval_path = Path("fixture/eval.jsonl")

    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=3, save_steps=100)
    cfg = enable_evaluation(cfg, dataset_refs=(_eval_ref(eval_path),))
    cfg = replace(
        cfg,
        export=replace(cfg.export, output_dir=export_dir),
    )
    ctx0 = build_stub_training_context(config=cfg)

    raw_config = {
        "training": {"backend": "stub", "max_steps": 3},
        "resume": {"policy": "strict"},
        "checkpointing": {"save_steps": 100, "save_total_limit": 3},
        "evaluation": {"enabled": True, "backend": "stub"},
        "export": {
            "enabled": True,
            "backend": "stub",
            "output_dir": str(export_dir),
        },
    }

    pipe = Pipeline(build_phase4_pipeline())
    result = pipe.run(
        PipelineContext(
            experiment_id=cfg.experiment_id,
            run_id=RunId(value="pipe-run"),
            config=cfg,
        ).with_values(
            raw_config=raw_config,
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
    assert statuses["evaluate"] is StageStatus.SUCCEEDED
    assert statuses["export"] is StageStatus.SUCCEEDED

    export_stage = next(r for r in result.stage_results if r.stage.value == "export")
    assert "bundle=" in (export_stage.message or "")
