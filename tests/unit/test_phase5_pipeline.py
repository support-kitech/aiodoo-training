"""Phase 5 pipeline + resume fingerprint compatibility."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase5
from aiodoo_training.domain.enums import StageStatus, TrainingStatus
from aiodoo_training.domain.identifiers import RunId
from aiodoo_training.pipeline import Pipeline, PipelineContext, build_phase4_pipeline
from aiodoo_training.pipeline.handlers import PlanCurriculumStage, PlanPackingStage
from aiodoo_training.training.engine import (
    build_stub_training_context,
    make_stub_experiment_config,
)
from tests.unit.phase5_helpers import make_examples, plan_once


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase5(overwrite=True)


def test_plan_packing_and_curriculum_stages_with_examples() -> None:
    cfg = make_stub_experiment_config(output_dir=Path("/tmp/p5"), max_steps=1, save_steps=100)
    examples = make_examples(4)
    ctx = PipelineContext(
        experiment_id=cfg.experiment_id,
        run_id=RunId(value="p5"),
        config=cfg,
    ).with_values(
        training_examples=examples,
        raw_config={
            "packing": {"backend": "concat", "mode": "concat", "max_sequence_length": 64},
            "curriculum": {
                "backend": "sequential",
                "mode": "sequential",
                "stages": ["easy", "hard"],
            },
            "sampling": {"backend": "identity", "seed": 42},
        },
    )
    ctx, result = PlanPackingStage().run(ctx)
    assert result.status is StageStatus.SUCCEEDED
    assert ctx.get("schedule_plan") is not None
    assert ctx.get("packing_statistics") is not None
    assert ctx.get("curriculum_statistics") is not None
    ctx2, result2 = PlanCurriculumStage().run(ctx)
    assert result2.status is StageStatus.SUCCEEDED
    assert ctx2.get("schedule_plan") is ctx.get("schedule_plan")


def test_plan_packing_idempotent_reuse() -> None:
    cfg = make_stub_experiment_config(output_dir=Path("/tmp/p5b"), max_steps=1, save_steps=100)
    examples = make_examples(3)
    ctx = PipelineContext(
        experiment_id=cfg.experiment_id,
        run_id=RunId(value="p5b"),
        config=cfg,
    ).with_values(training_examples=examples, raw_config={})
    ctx, _ = PlanPackingStage().run(ctx)
    plan = ctx.get("schedule_plan")
    ctx2, result = PlanPackingStage().run(ctx)
    assert result.message and "reused" in result.message
    assert ctx2.get("schedule_plan") is plan


def test_phase4_pipeline_still_completes(tmp_path: Path) -> None:
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=2, save_steps=100)
    ctx0 = build_stub_training_context(config=cfg)
    pipe = Pipeline(build_phase4_pipeline())
    result = pipe.run(
        PipelineContext(
            experiment_id=cfg.experiment_id,
            run_id=RunId(value="pipe-p5"),
            config=cfg,
        ).with_values(
            raw_config={
                "training": {"backend": "stub", "max_steps": 2},
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


def test_resume_fingerprints_stable_across_plans() -> None:
    a = plan_once(seed=7)
    b = plan_once(seed=7)
    # Resume compatibility: packing/curriculum fingerprints are deterministic
    # and can be folded into checkpoint / config fingerprint material.
    assert a.packing_fingerprint == b.packing_fingerprint
    assert a.curriculum_fingerprint == b.curriculum_fingerprint
    assert a.sampling_fingerprint == b.sampling_fingerprint
    assert a.packing_session.packing_fingerprint == a.packing_fingerprint
    assert a.curriculum_session.curriculum_fingerprint == a.curriculum_fingerprint
