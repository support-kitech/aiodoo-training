"""Optional HuggingFace CheckpointStore — graceful without transformers."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from aiodoo_training.domain.enums import CheckpointType
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.training import CheckpointHandle, TrainingProgress
from aiodoo_training.exceptions import DomainError, FactoryError
from aiodoo_training.ports.trainer import CheckpointStore


class HFCheckpointStore(CheckpointStore):
    """
    HuggingFace / PEFT weight package store.

    Registered for selection when HF training extras are present. Without
    transformers (Phase 3 default CI), save/restore raise a clear error so
    callers fall back to the stub store.
    """

    BACKEND_KEY = "hf"

    def save(
        self,
        model: TrainableModelHandle,
        progress: TrainingProgress,
        experiment_id: ExperimentId,
        run_id: RunId,
        destination: Path,
    ) -> CheckpointHandle:
        self._require_transformers()
        framework = getattr(model, "framework_model", None)
        if framework is None or (isinstance(framework, dict) and framework.get("kind") == "stub"):
            raise DomainError(
                "HFCheckpointStore cannot persist stub models; use CheckpointStore key 'stub'."
            )
        destination.mkdir(parents=True, exist_ok=True)
        # Thin path: prefer PEFT / transformers save_pretrained when available.
        save_pretrained = getattr(framework, "save_pretrained", None)
        if callable(save_pretrained):
            save_pretrained(str(destination))
        else:
            raise DomainError(
                "HFCheckpointStore requires a transformers/peft model exposing save_pretrained."
            )
        return CheckpointHandle(
            path=destination,
            experiment_id=experiment_id,
            run_id=run_id,
            checkpoint_type=CheckpointType.FULL_STATE,
            global_step=progress.global_step,
            created_at=datetime.now(UTC),
            metadata=(("store", self.BACKEND_KEY),),
        )

    def restore(self, handle: CheckpointHandle) -> TrainableModelHandle:
        self._require_transformers()
        raise DomainError(
            "HFCheckpointStore.restore requires application-level ModelLoader + "
            "AdaptationApplier rehydration in Phase 3; use StubCheckpointStore for "
            f"CPU CI. Checkpoint path: {handle.path}"
        )

    def list(self, directory: Path) -> Sequence[CheckpointHandle]:
        if not directory.is_dir():
            return ()
        handles: list[CheckpointHandle] = []
        for child in sorted(directory.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or not child.name.startswith("checkpoint-"):
                continue
            step = 0
            suffix = child.name.removeprefix("checkpoint-")
            try:
                step = int(suffix)
            except ValueError:
                step = 0
            handles.append(
                CheckpointHandle(
                    path=child,
                    experiment_id=ExperimentId(value="unknown"),
                    run_id=RunId(value="unknown"),
                    checkpoint_type=CheckpointType.FULL_STATE,
                    global_step=step,
                    created_at=None,
                    metadata=(("store", self.BACKEND_KEY),),
                )
            )
        return tuple(handles)

    def prune(self, directory: Path, keep: int) -> Sequence[CheckpointHandle]:
        if keep < 0:
            raise DomainError("prune keep must be >= 0.")
        handles = list(self.list(directory))
        if len(handles) <= keep:
            return ()
        removed = handles[: len(handles) - keep] if keep > 0 else handles
        for handle in removed:
            if handle.path.is_dir():
                shutil.rmtree(handle.path, ignore_errors=True)
        return tuple(removed)

    @staticmethod
    def _require_transformers() -> None:
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            raise FactoryError(
                "HFCheckpointStore requires the 'transformers' package. "
                "Use CheckpointStore key 'stub' for CPU CI."
            ) from exc


def register_hf_checkpoint_store(*, overwrite: bool = False) -> None:
    from aiodoo_training.registries import checkpoint_store_registry

    if not checkpoint_store_registry.exists("hf") or overwrite:
        checkpoint_store_registry.register("hf", HFCheckpointStore, overwrite=overwrite)
    if not checkpoint_store_registry.exists("huggingface") or overwrite:
        checkpoint_store_registry.register(
            "huggingface", HFCheckpointStore, overwrite=overwrite
        )
