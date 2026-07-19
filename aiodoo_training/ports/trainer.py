"""Trainer / checkpoint / eval / export / tracking / RNG ports."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from aiodoo_training.domain.artifacts import EvaluationReport, ExportArtifact
from aiodoo_training.domain.config import EvaluationSpec, ExperimentConfig, ExportSpec
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.domain.training import CheckpointHandle, MetricSnapshot, TrainingProgress


class TrainerBackend(ABC):
    """Executes or resumes a training loop behind a stable interface."""

    @abstractmethod
    def train(
        self,
        config: ExperimentConfig,
        model: TrainableModelHandle,
        execution: ExecutionEnvironment,
    ) -> TrainingProgress:
        """Run training from scratch for the given experiment configuration."""

    @abstractmethod
    def resume(
        self,
        config: ExperimentConfig,
        model: TrainableModelHandle,
        checkpoint: CheckpointHandle,
        execution: ExecutionEnvironment,
    ) -> TrainingProgress:
        """Resume training from a previously persisted checkpoint."""


class CheckpointStore(ABC):
    """Persists and restores training state atomically."""

    @abstractmethod
    def save(
        self,
        model: TrainableModelHandle,
        progress: TrainingProgress,
        experiment_id: ExperimentId,
        run_id: RunId,
        destination: Path,
    ) -> CheckpointHandle:
        """Persist checkpoint state and return a handle."""

    @abstractmethod
    def restore(self, handle: CheckpointHandle) -> TrainableModelHandle:
        """Restore model/optimizer state from a checkpoint handle."""

    @abstractmethod
    def list(self, directory: Path) -> Sequence[CheckpointHandle]:
        """List known checkpoints under a directory."""

    @abstractmethod
    def prune(self, directory: Path, keep: int) -> Sequence[CheckpointHandle]:
        """Prune old checkpoints, retaining at most ``keep`` entries."""


class Evaluator(ABC):
    """Runs offline evaluation against held-out datasets."""

    @abstractmethod
    def evaluate(
        self,
        model: TrainableModelHandle,
        dataset_refs: Sequence[DatasetRef],
        spec: EvaluationSpec,
        experiment_id: ExperimentId,
        run_id: RunId,
        execution: ExecutionEnvironment,
    ) -> EvaluationReport:
        """Evaluate the model and return an immutable report."""


class Exporter(ABC):
    """Exports trained artifacts for Capability Package / ArtifactBundle consumers."""

    @abstractmethod
    def export(
        self,
        model: TrainableModelHandle,
        spec: ExportSpec,
        experiment_id: ExperimentId,
        run_id: RunId,
    ) -> Sequence[ExportArtifact]:
        """Produce one or more export artifacts."""


class ExperimentTracker(ABC):
    """Records parameters, metrics, and artifact references for a run."""

    @abstractmethod
    def log_params(self, params: dict[str, object]) -> None:
        """Log static run parameters."""

    @abstractmethod
    def log_metrics(self, metrics: Sequence[MetricSnapshot]) -> None:
        """Log metric snapshots."""

    @abstractmethod
    def log_artifact(self, path: Path, name: str | None = None) -> None:
        """Attach an artifact path to the run."""

    @abstractmethod
    def close(self) -> None:
        """Flush and close the tracking sink."""


class RngController(ABC):
    """Controls seeding and RNG state for reproducible experiments."""

    @abstractmethod
    def seed_all(self, seed: int) -> None:
        """Seed all relevant RNGs for the process."""

    @abstractmethod
    def snapshot(self) -> dict[str, object]:
        """Capture RNG state for checkpointing."""

    @abstractmethod
    def restore(self, state: dict[str, object]) -> None:
        """Restore RNG state from a snapshot."""
