"""Pipeline construction and orchestration tests."""

import pytest

from aiodoo_training.domain.enums import PipelineStage, StageStatus, TrainingStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId, StageName
from aiodoo_training.domain.pipeline import StageResult
from aiodoo_training.exceptions import PipelineError
from aiodoo_training.pipeline import NoOpStage, Pipeline, PipelineContext, PipelineStageHandler


class FailingStage(PipelineStageHandler):
    @property
    def name(self) -> StageName:
        return StageName(value="fail")

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.TRAIN

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        return context, StageResult(
            name=self.name,
            stage=self.stage,
            status=StageStatus.FAILED,
            message="boom",
            error="intentional",
        )


class RaisingStage(PipelineStageHandler):
    @property
    def name(self) -> StageName:
        return StageName(value="raise")

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.EVALUATE

    def run(self, context: PipelineContext) -> tuple[PipelineContext, StageResult]:
        raise RuntimeError("unexpected")


def test_empty_pipeline_completes() -> None:
    result = Pipeline().run(
        PipelineContext(
            experiment_id=ExperimentId(value="exp_1"),
            run_id=RunId(value="run_1"),
        )
    )
    assert result.status == TrainingStatus.COMPLETED
    assert result.succeeded
    assert result.stage_results == ()


def test_noop_stages_run_in_order() -> None:
    pipeline = Pipeline(
        [
            NoOpStage("validate", PipelineStage.VALIDATE_CONFIG),
            NoOpStage("finalize", PipelineStage.FINALIZE),
        ]
    )
    result = pipeline.run()
    assert result.status == TrainingStatus.COMPLETED
    assert [s.name.value for s in result.stage_results] == ["validate", "finalize"]


def test_pipeline_stops_on_failure_and_exception() -> None:
    failed = Pipeline(
        [
            NoOpStage("ok", PipelineStage.VALIDATE_CONFIG),
            FailingStage(),
            NoOpStage("never", PipelineStage.FINALIZE),
        ]
    ).run()
    assert failed.status == TrainingStatus.FAILED
    assert len(failed.stage_results) == 2

    raised = Pipeline([RaisingStage()]).run()
    assert raised.status == TrainingStatus.FAILED
    assert "RuntimeError" in (raised.stage_results[0].error or "")


def test_context_copy_on_write_and_identity() -> None:
    ctx = PipelineContext()
    updated = ctx.with_values(answer=42)
    assert ctx.get("answer") is None
    assert updated.get("answer") == 42

    identified = ctx.with_identity(experiment_id=ExperimentId(value="exp_x"))
    assert identified.experiment_id is not None
    assert identified.experiment_id.value == "exp_x"


def test_cannot_add_stage_after_run_starts() -> None:
    pipeline = Pipeline()
    pipeline.run()
    with pytest.raises(PipelineError, match="after Pipeline.run"):
        pipeline.add_stage(NoOpStage("late", PipelineStage.FINALIZE))
