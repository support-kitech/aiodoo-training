"""Export runtime context — resolved application bag for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from aiodoo_training.domain.artifacts import EvaluationReport, ExportArtifact
from aiodoo_training.domain.config import ExperimentConfig, ExportSpec
from aiodoo_training.domain.export_manifest import (
    ArtifactCompatibilityPolicy,
    ArtifactValidationPolicy,
)
from aiodoo_training.domain.export_session import ExportSession
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.quality import QualityReport
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.ports.trainer import ExperimentTracker, Exporter


@dataclass(frozen=True, slots=True)
class ExportContext:
    """
    Resolved runtime collaborators for export pipeline stages.

    Built by builders; consumed by ExportManager and bindable for ``Exporter``
    infrastructure adapters — never widens frozen port signatures.
    """

    config: ExperimentConfig
    export_spec: ExportSpec
    model: TrainableModelHandle
    execution: ExecutionEnvironment
    export_session: ExportSession
    exporter: Exporter
    output_dir: Path
    tmp_dir: Path
    exporter_backend_key: str = "stub"
    model_fingerprint: str = ""
    adapter_fingerprint: str = ""
    config_fingerprint: str = ""
    evaluation_fingerprint: str | None = None
    evaluation_report: EvaluationReport | None = None
    quality_report: QualityReport | None = None
    validation_policy: ArtifactValidationPolicy = ArtifactValidationPolicy.STRICT
    compatibility_policy: ArtifactCompatibilityPolicy | None = None
    require_evaluation: bool = False
    require_pass_for_export: bool = False
    export_types: tuple[str, ...] = ("peft_adapter", "tokenizer", "manifest", "bundle")
    tracker: ExperimentTracker | None = None
    artifacts: tuple[ExportArtifact, ...] = ()
    bind_extra: dict[str, Any] = field(default_factory=dict)

    def with_export_session(self, session: ExportSession) -> ExportContext:
        return replace(self, export_session=session)

    def with_tmp_dir(self, tmp_dir: Path) -> ExportContext:
        return replace(self, tmp_dir=tmp_dir)

    def with_artifacts(self, artifacts: tuple[ExportArtifact, ...]) -> ExportContext:
        return replace(self, artifacts=artifacts)
