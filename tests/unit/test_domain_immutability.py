"""Immutability tests for domain objects."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from aiodoo_training.domain import (
    AdapterType,
    DatasetRef,
    DatasetType,
    ExperimentId,
    ModelFamily,
    ModelRef,
    PipelineResult,
    Precision,
    RunId,
    StageName,
    StageResult,
    TrainingStatus,
)
from aiodoo_training.domain.enums import PipelineStage, StageStatus


def test_experiment_id_immutable() -> None:
    eid = ExperimentId(value="exp_abc")
    with pytest.raises(FrozenInstanceError):
        eid.value = "other"  # type: ignore[misc]


def test_empty_experiment_id_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentId(value="")


def test_refs_immutable() -> None:
    model = ModelRef(identifier="m", family=ModelFamily.QWEN, precision=Precision.BF16)
    dataset = DatasetRef(
        path=Path("data.jsonl"),
        dataset_type=DatasetType.CODING,
        protocol_version="1.0",
    )
    with pytest.raises(FrozenInstanceError):
        model.identifier = "x"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        dataset.dataset_type = DatasetType.PLANNER  # type: ignore[misc]


def test_pipeline_result_succeeded_property() -> None:
    result = PipelineResult(
        experiment_id=ExperimentId(value="exp_1"),
        run_id=RunId(value="run_1"),
        status=TrainingStatus.COMPLETED,
        stage_results=(
            StageResult(
                name=StageName(value="noop"),
                stage=PipelineStage.FINALIZE,
                status=StageStatus.SUCCEEDED,
            ),
        ),
    )
    assert result.succeeded is True
    assert AdapterType.LORA.value == "lora"
