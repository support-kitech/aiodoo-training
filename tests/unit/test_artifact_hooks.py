"""Unit tests for artifact publish hooks."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aiodoo_training.artifacts.publish_contract import PublishError
from aiodoo_training.domain.artifacts import EvaluationReport
from aiodoo_training.domain.enums import TrainingStatus
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.quality import QualityReport
from aiodoo_training.domain.training import TrainingProgress
from aiodoo_training.pipeline.artifact_hooks import maybe_publish_artifacts
from aiodoo_training.pipeline.pipeline import PipelineContext


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "AIODOO"
    root.mkdir()
    return root


@pytest.fixture
def resolved_config(workspace: Path) -> dict:
    return {
        "name": "EXP-0001",
        "experiment": {"id": "EXP-0001"},
        "workspace": {"layout": "drive_v1", "root": str(workspace)},
        "dataset_version": "v1.0.0",
    }


def _completed_progress() -> TrainingProgress:
    return TrainingProgress(status=TrainingStatus.COMPLETED, global_step=10, epoch=1.0)


def test_summary_success_true_when_evaluation_skipped(
    workspace: Path, resolved_config: dict
) -> None:
    context = PipelineContext(run_id=RunId(value="run-1")).with_values(
        raw_config=resolved_config,
        training_progress=_completed_progress(),
    )
    maybe_publish_artifacts(context)
    summary_path = workspace / "experiments" / "EXP-0001" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["success"] is True


def test_summary_success_false_when_evaluation_failed(
    workspace: Path, resolved_config: dict
) -> None:
    evaluation_report = EvaluationReport(
        experiment_id=ExperimentId(value="EXP-0001"),
        run_id=RunId(value="run-1"),
        metrics=(),
        passed=False,
        details="threshold exceeded",
    )
    context = PipelineContext(run_id=RunId(value="run-1")).with_values(
        raw_config=resolved_config,
        training_progress=_completed_progress(),
        evaluation_report=evaluation_report,
    )
    maybe_publish_artifacts(context)
    summary = json.loads(
        (workspace / "experiments" / "EXP-0001" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["success"] is False


def test_summary_success_false_when_quality_gate_failed(
    workspace: Path, resolved_config: dict
) -> None:
    evaluation_report = EvaluationReport(
        experiment_id=ExperimentId(value="EXP-0001"),
        run_id=RunId(value="run-1"),
        metrics=(),
        passed=True,
    )
    quality_report = QualityReport(passed=False, details="gate failed")
    context = PipelineContext(run_id=RunId(value="run-1")).with_values(
        raw_config=resolved_config,
        training_progress=_completed_progress(),
        evaluation_report=evaluation_report,
        quality_report=quality_report,
    )
    maybe_publish_artifacts(context)
    summary = json.loads(
        (workspace / "experiments" / "EXP-0001" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["success"] is False


def test_publish_error_is_logged_not_raised(
    workspace: Path, resolved_config: dict, caplog: pytest.LogCaptureFixture
) -> None:
    ckpt = workspace / "training" / "cache" / "EXP-0001" / "checkpoints" / "checkpoint-1"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")

    context = PipelineContext(run_id=RunId(value="run-1")).with_values(
        raw_config=resolved_config,
        training_progress=_completed_progress(),
    )

    with patch(
        "aiodoo_training.pipeline.artifact_hooks.ArtifactOutputManager.publish_adapter_from_checkpoint",
        side_effect=PublishError("invalid checkpoint"),
    ):
        with caplog.at_level("WARNING"):
            maybe_publish_artifacts(context)

    assert any("Adapter publish failed" in record.message for record in caplog.records)
    summary = json.loads(
        (workspace / "experiments" / "EXP-0001" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["success"] is True


def test_warns_when_no_checkpoint_to_publish(
    workspace: Path, resolved_config: dict, caplog: pytest.LogCaptureFixture
) -> None:
    context = PipelineContext(run_id=RunId(value="run-1")).with_values(
        raw_config=resolved_config,
        training_progress=_completed_progress(),
    )
    with caplog.at_level("WARNING"):
        maybe_publish_artifacts(context)

    assert any("No checkpoint" in record.message for record in caplog.records)
    summary = json.loads(
        (workspace / "experiments" / "EXP-0001" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["paths"]["adapter"] is None


def test_smoke_md_validation_command_uses_odoo_versions_flag() -> None:
    smoke_doc = Path(__file__).resolve().parents[2] / "docs" / "SMOKE.md"
    text = smoke_doc.read_text(encoding="utf-8")
    assert "--odoo-versions 18" in text
    assert "--odoo-version 18" not in text
