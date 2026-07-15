"""Thin HuggingFace Exporter — optional transformers dependency."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiodoo_training.domain.artifacts import ExportArtifact
from aiodoo_training.domain.config import ExportSpec
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.exceptions import DomainError
from aiodoo_training.infrastructure.stub.exporter import StubExporter
from aiodoo_training.ports.trainer import Exporter

if TYPE_CHECKING:
    from aiodoo_training.export.context import ExportContext
else:
    ExportContext = Any  # type: ignore[misc,assignment]


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise DomainError(
            "HFExporter requires the 'transformers' package. "
            "Install training extras or use Exporter key 'stub' for CPU CI."
        ) from exc
    return transformers


class HFExporter(Exporter):
    """
    Phase 4 HuggingFace export adapter.

    Registered without importing transformers at module load time.
    Phase 4 delegates to stub exporter for deterministic bundle layout.
    """

    BACKEND_KEY = "hf_peft"

    def __init__(self, context: ExportContext | None = None) -> None:
        self._context = context
        self._stub = StubExporter(context)

    def bind(self, context: ExportContext) -> HFExporter:
        self._context = context
        self._stub.bind(context)
        return self

    @property
    def context(self) -> ExportContext | None:
        return self._context

    def export(
        self,
        model: TrainableModelHandle,
        spec: ExportSpec,
        experiment_id: ExperimentId,
        run_id: RunId,
    ) -> tuple[ExportArtifact, ...]:
        try:
            _require_transformers()
        except DomainError:
            return self._stub.export(model, spec, experiment_id, run_id)
        return self._stub.export(model, spec, experiment_id, run_id)


def register_hf_exporter(*, overwrite: bool = False) -> None:
    from aiodoo_training.registries import exporter_registry

    if not exporter_registry.exists("hf_peft") or overwrite:
        exporter_registry.register("hf_peft", HFExporter, overwrite=overwrite)
