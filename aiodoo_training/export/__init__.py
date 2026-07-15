"""Phase 4 export application layer."""

from aiodoo_training.export.context import ExportContext
from aiodoo_training.export.fingerprints import compute_export_fingerprint
from aiodoo_training.export.index import ArtifactIndex
from aiodoo_training.export.lifecycle import ExportLifecycle
from aiodoo_training.export.manager import ExportManager
from aiodoo_training.export.model_card import ModelCardBuilder

__all__ = [
    "ArtifactIndex",
    "ExportContext",
    "ExportLifecycle",
    "ExportManager",
    "ModelCardBuilder",
    "build_stub_export_context",
    "compute_export_fingerprint",
    "run_stub_export",
]


def __getattr__(name: str) -> object:
    if name == "build_stub_export_context":
        from aiodoo_training.export.harness import build_stub_export_context

        return build_stub_export_context
    if name == "run_stub_export":
        from aiodoo_training.export.harness import run_stub_export

        return run_stub_export
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
