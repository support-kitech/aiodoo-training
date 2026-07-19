"""Evaluation and export artifact domain objects."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aiodoo_training.domain.enums import ExportType
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.training import MetricSnapshot


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Result of an offline evaluation pass."""

    experiment_id: ExperimentId
    run_id: RunId
    metrics: tuple[MetricSnapshot, ...]
    passed: bool
    details: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ExportArtifact:
    """A single exported artifact role inside an ArtifactBundle."""

    export_type: ExportType
    path: Path
    checksum: str | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """
    Manifest describing a completed (or exportable) training experiment.

    Provenance for export / Capability Package handoff (not a registry document).
    """

    experiment_id: ExperimentId
    run_id: RunId
    name: str
    config_hash: str
    artifacts: tuple[ExportArtifact, ...] = ()
    metrics: tuple[MetricSnapshot, ...] = ()
    created_at: datetime | None = None
