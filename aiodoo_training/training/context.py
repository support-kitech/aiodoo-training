"""Training runtime context — resolved application bag for pipeline stages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aiodoo_training.domain.config import ExperimentConfig
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.domain.training_policies import (
    CheckpointPolicy,
    GradientAccumulationPolicy,
    GradientClippingPolicy,
    LossScalingPolicy,
    MixedPrecisionPolicy,
    OptimizerPolicy,
    SchedulerPolicy,
)
from aiodoo_training.domain.training_session import TrainingSession
from aiodoo_training.ports.callback import TrainingCallback
from aiodoo_training.ports.trainer import (
    CheckpointStore,
    ExperimentTracker,
    RngController,
    TrainerBackend,
)

if TYPE_CHECKING:
    from aiodoo_training.training.checkpoint_manager import CheckpointManager
    from aiodoo_training.training.event_bus import TrainingEventBus
    from aiodoo_training.training.metrics import MetricCollector, TrainingHistory


@dataclass(frozen=True, slots=True)
class TrainingContext:
    """
    Resolved runtime collaborators for training pipeline stages.

    Built by builders; consumed by stages and bindable for ``TrainerBackend``
    infrastructure adapters via opaque ``extra`` metadata — never widens frozen
    port signatures.
    """

    config: ExperimentConfig
    execution: ExecutionEnvironment
    model: TrainableModelHandle
    dataset_session: DatasetSession
    training_session: TrainingSession
    trainer: TrainerBackend
    checkpoint_store: CheckpointStore
    rng: RngController
    optimizer_policy: OptimizerPolicy
    scheduler_policy: SchedulerPolicy
    gradient_accumulation_policy: GradientAccumulationPolicy
    gradient_clipping_policy: GradientClippingPolicy
    mixed_precision_policy: MixedPrecisionPolicy
    loss_scaling_policy: LossScalingPolicy
    checkpoint_policy: CheckpointPolicy
    callbacks: tuple[TrainingCallback, ...] = ()
    tracker: ExperimentTracker | None = None
    event_bus: TrainingEventBus | None = None
    checkpoint_manager: CheckpointManager | None = None
    metric_collector: MetricCollector | None = None
    training_history: TrainingHistory | None = None
    trainer_backend_key: str = "hf_trainer"
    model_fingerprint: str = ""
    adapter_fingerprint: str = ""
    config_fingerprint: str = ""
    execution_digest: str = ""
    quantization_digest: str = ""
    adaptation_strategy_key: str = ""
    bind_extra: dict[str, Any] = field(default_factory=dict)

    def with_training_session(self, session: TrainingSession) -> TrainingContext:
        from dataclasses import replace

        return replace(self, training_session=session)

    def with_dataset_session(self, dataset_session: DatasetSession) -> TrainingContext:
        from dataclasses import replace

        return replace(self, dataset_session=dataset_session)

    def with_model(self, model: TrainableModelHandle) -> TrainingContext:
        from dataclasses import replace

        return replace(self, model=model)


@dataclass(frozen=True, slots=True)
class CallbackContext:
    """
    Limited API surface exposed to :class:`TrainingCallback` listeners.

    Callbacks must not mutate :class:`TrainingSession` in place; they may
    request manager actions through this façade.
    """

    training_context: TrainingContext

    @property
    def config(self) -> ExperimentConfig:
        return self.training_context.config

    @property
    def training_session(self) -> TrainingSession:
        return self.training_context.training_session

    @property
    def dataset_session(self) -> DatasetSession:
        return self.training_context.dataset_session

    @property
    def execution(self) -> ExecutionEnvironment:
        return self.training_context.execution

    @property
    def checkpoint_manager(self) -> CheckpointManager | None:
        return self.training_context.checkpoint_manager

    @property
    def metric_collector(self) -> MetricCollector | None:
        return self.training_context.metric_collector

    @property
    def training_history(self) -> TrainingHistory | None:
        return self.training_context.training_history

    @property
    def callbacks(self) -> Sequence[TrainingCallback]:
        return self.training_context.callbacks
