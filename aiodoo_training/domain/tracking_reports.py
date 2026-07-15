"""Phase 6 reporting / summary domain DTOs (derived views)."""

from __future__ import annotations

from dataclasses import dataclass

from aiodoo_training.domain.enums import RunState
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.tracking_policies import TRACKING_PROTOCOL_VERSION
from aiodoo_training.domain.training import MetricSnapshot


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """Derived training view for humans / JSON reports."""

    experiment_id: ExperimentId
    run_id: RunId
    global_step: int = 0
    epoch: float = 0.0
    status: str = ""
    metrics: tuple[MetricSnapshot, ...] = ()
    checkpoint_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationReportSummary:
    """Projection of frozen EvaluationReport (not a redesign)."""

    experiment_id: ExperimentId
    run_id: RunId
    metric_names: tuple[str, ...] = ()
    metric_values: tuple[float, ...] = ()
    passed: bool | None = None


@dataclass(frozen=True, slots=True)
class ExportReport:
    """Derived export view from ExportSession + artifact pointers."""

    experiment_id: ExperimentId
    run_id: RunId
    artifact_refs: tuple[str, ...] = ()
    export_status: str = ""


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    """Catalog list row for an experiment."""

    experiment_id: ExperimentId
    name: str
    status: str
    run_count: int = 0
    latest_run_id: RunId | None = None
    config_fingerprint: str = ""
    tracking_protocol_version: str = TRACKING_PROTOCOL_VERSION


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Catalog list row for a run."""

    run_id: RunId
    experiment_id: ExperimentId
    state: RunState
    provenance_digest: str = ""
    packing_fingerprint: str | None = None
    curriculum_fingerprint: str | None = None
    tracking_protocol_version: str = TRACKING_PROTOCOL_VERSION


@dataclass(frozen=True, slots=True)
class ExportStatistics:
    """Immutable completed-export summary (recording only)."""

    export_fingerprint: str
    artifact_count: int
    roles: tuple[str, ...] = ()
