"""Checkpoint resume bundle assembly."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from aiodoo_training.domain.checkpoint_manifest import CheckpointManifest
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.domain.training import CheckpointHandle
from aiodoo_training.domain.training_policies import ResumePolicy
from aiodoo_training.domain.training_session import TrainingSession
from aiodoo_training.ports.trainer import CheckpointStore, RngController
from aiodoo_training.training.checkpoint_manager import (
    CheckpointManager,
    ResumeValidationContext,
    dataset_session_from_dict,
)


@dataclass(frozen=True, slots=True)
class ResumeBundle:
    """Validated artifacts required to continue training from a checkpoint."""

    model: TrainableModelHandle
    checkpoint: CheckpointHandle
    training_session: TrainingSession
    dataset_session: DatasetSession
    rng_state: dict[str, object]
    manifest: CheckpointManifest
    warnings: tuple[str, ...] = ()


class ResumeCoordinator:
    """
    Application coordinator for checkpoint resume.

    Delegates validation and sidecar restoration to :class:`CheckpointManager`.
    """

    def __init__(
        self,
        *,
        checkpoint_store: CheckpointStore,
        rng: RngController,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        self._store = checkpoint_store
        self._rng = rng
        self._manager = checkpoint_manager or CheckpointManager(
            checkpoint_store=checkpoint_store,
            rng=rng,
        )

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        return self._manager

    def load_and_validate(
        self,
        path: Path,
        *,
        expected: ResumeValidationContext,
        policy: ResumePolicy = ResumePolicy.STRICT,
        training_session: TrainingSession,
    ) -> ResumeBundle:
        return self._manager.load_and_validate(
            path,
            expected=expected,
            policy=policy,
            training_session=training_session,
        )

    def apply_rng(self, bundle: ResumeBundle) -> None:
        """Restore RNG state captured in the resume bundle."""
        self._rng.restore(bundle.rng_state)

    def rebuild_training_session(
        self,
        base: TrainingSession,
        manifest: CheckpointManifest,
        *,
        checkpoint_path: Path | None = None,
        warnings: tuple[str, ...] = (),
    ) -> TrainingSession:
        """Merge manifest progress fields into a copy of ``base``."""
        meta = dict(base.metadata)
        for index, warning in enumerate(warnings):
            meta[f"resume_warning_{index}"] = warning
        dataset_session = dataset_session_from_dict(manifest.dataset_session)
        return replace(
            base,
            global_step=manifest.global_step,
            epoch=manifest.epoch,
            checkpoint_fingerprint=manifest.checkpoint_fingerprint,
            resume_from=checkpoint_path if checkpoint_path is not None else base.resume_from,
            dataset_session=dataset_session,
            metadata=MappingProxyType(meta),
        )
