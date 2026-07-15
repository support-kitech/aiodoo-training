"""Report rendering — derived JSON views."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from aiodoo_training.domain.tracking_reports import (
    EvaluationReportSummary,
    ExperimentSummary,
    ExportReport,
    RunSummary,
    TrainingReport,
)


def _to_dict(obj: object) -> dict[str, Any]:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Unsupported report type: {type(obj)!r}")


class ReportRenderer:
    """Renders summary DTOs to JSON text / files."""

    def render(self, summary: object) -> str:
        return json.dumps(_to_dict(summary), sort_keys=True, indent=2, default=str) + "\n"

    def write(self, summary: object, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(summary), encoding="utf-8")
        return path


def write_run_reports(
    *,
    run_dir: Path,
    training: TrainingReport | None = None,
    evaluation: EvaluationReportSummary | None = None,
    export: ExportReport | None = None,
) -> None:
    renderer = ReportRenderer()
    reports = run_dir / "reports"
    if training is not None:
        renderer.write(training, reports / "training.json")
    if evaluation is not None:
        renderer.write(evaluation, reports / "evaluation.json")
    if export is not None:
        renderer.write(export, reports / "export.json")


__all__ = [
    "EvaluationReportSummary",
    "ExperimentSummary",
    "ExportReport",
    "ReportRenderer",
    "RunSummary",
    "TrainingReport",
    "write_run_reports",
]
