"""Immutable configuration domain objects (placeholders for Phase 0)."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from aiodoo_training.domain.enums import (
    AcceleratorKind,
    AdapterType,
    CurriculumMode,
    PackingMode,
    Precision,
    TrackerType,
    TrainingBackend,
)
from aiodoo_training.domain.identifiers import ExperimentId
from aiodoo_training.domain.refs import AdapterRef, DatasetRef, ModelRef
from aiodoo_training.domain.resources import DevicePolicy, MemoryPolicy, PrecisionPolicy


@dataclass(frozen=True, slots=True)
class DatasetMixSpec:
    """Weighted mixture of dataset references."""

    datasets: tuple[DatasetRef, ...] = ()
    shuffle: bool = True
    seed: int = 42


@dataclass(frozen=True, slots=True)
class AdaptationSpec:
    """High-level adaptation policy (LoRA / QLoRA / full)."""

    adapter_type: AdapterType = AdapterType.LORA
    rank: int | None = None
    alpha: int | None = None
    dropout: float | None = None
    target_modules: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    """Optimizer and schedule hyperparameters."""

    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_epochs: float = 1.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int | None = None


@dataclass(frozen=True, slots=True)
class PrecisionSpec:
    """Training-loop precision policy (AMP / dtype for the trainer backend)."""

    precision: Precision = Precision.BF16
    gradient_checkpointing: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """
    Declared execution preferences resolved into an ExecutionEnvironment.

    Future-compatible for multi-GPU, Apple Silicon, DeepSpeed, and FSDP without
    schema redesign — accelerators and device ids are policy fields only.
    """

    device: DevicePolicy = field(default_factory=DevicePolicy)
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)
    memory: MemoryPolicy = field(default_factory=MemoryPolicy)
    accelerator: AcceleratorKind = AcceleratorKind.NONE


@dataclass(frozen=True, slots=True)
class DistributedSpec:
    """
    Declared distributed topology (inactive until Phase 7+).

    Values are informational for fingerprints and DatasetSession placement.
    """

    enabled: bool = False
    world_size: int = 1
    global_rank: int = 0
    local_rank: int = 0
    num_nodes: int = 1

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("DistributedSpec.world_size must be >= 1.")
        if self.num_nodes < 1:
            raise ValueError("DistributedSpec.num_nodes must be >= 1.")


@dataclass(frozen=True, slots=True)
class PackingSpec:
    """Sequence packing configuration."""

    mode: PackingMode = PackingMode.NONE
    max_sequence_length: int = 2048


@dataclass(frozen=True, slots=True)
class CurriculumSpec:
    """Curriculum scheduling configuration."""

    mode: CurriculumMode = CurriculumMode.NONE
    stages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CheckpointingSpec:
    """Checkpoint persistence policy."""

    output_dir: Path = field(default_factory=lambda: Path("artifacts/checkpoints"))
    save_steps: int = 500
    save_total_limit: int = 3
    resume_from: Path | None = None


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """Offline evaluation policy."""

    enabled: bool = False
    dataset_refs: tuple[DatasetRef, ...] = ()
    eval_steps: int | None = None


@dataclass(frozen=True, slots=True)
class ExportSpec:
    """Export policy for trained artifacts."""

    output_dir: Path = field(default_factory=lambda: Path("artifacts/export"))
    export_types: tuple[str, ...] = ("peft_adapter", "manifest")


@dataclass(frozen=True, slots=True)
class TrackingSpec:
    """Experiment tracking sink configuration."""

    tracker_type: TrackerType = TrackerType.NULL
    experiment_name: str | None = None
    tracking_uri: str | None = None


@dataclass(frozen=True, slots=True)
class DeterminismSpec:
    """Determinism controls for reproducible runs."""

    seed: int = 42
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """
    Fully composed, immutable experiment definition.

    Produced by the configuration system after load + compose + validate + resolve.
    """

    name: str
    schema_version: str
    seed: int
    model: ModelRef
    datasets: DatasetMixSpec
    adaptation: AdaptationSpec
    optimization: OptimizationSpec
    precision: PrecisionSpec
    packing: PackingSpec
    curriculum: CurriculumSpec
    checkpointing: CheckpointingSpec
    evaluation: EvaluationSpec
    export: ExportSpec
    tracking: TrackingSpec
    determinism: DeterminismSpec
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    distributed: DistributedSpec = field(default_factory=DistributedSpec)
    training_backend: TrainingBackend = TrainingBackend.HF_TRAINER
    adapter: AdapterRef | None = None
    experiment_id: ExperimentId | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
