"""Factory skeletons that resolve concrete ports from registries.

Factories depend on registries only. They must not import HuggingFace, PEFT,
or other infrastructure directly. Concrete backends are registered later and
constructed here (same pattern as aiodoo-datasets factories).
"""

from __future__ import annotations

from aiodoo_training.exceptions import FactoryError
from aiodoo_training.ports.callback import TrainingCallback
from aiodoo_training.ports.dataset import DatasetSource
from aiodoo_training.ports.distributed import (
    DistributedBackend,
    DistributedSampler,
    PlacementStrategy,
)
from aiodoo_training.ports.model import AdaptationStrategy, ModelBackend
from aiodoo_training.ports.optimizer import OptimizerBackend
from aiodoo_training.ports.packing import CurriculumStrategy, PackingStrategy, SamplingStrategy
from aiodoo_training.ports.resources import ResourcePlanner
from aiodoo_training.ports.scheduler import SchedulerBackend
from aiodoo_training.ports.tokenizer import TokenizerPort
from aiodoo_training.ports.trainer import (
    CheckpointStore,
    Evaluator,
    ExperimentTracker,
    Exporter,
    RngController,
    TrainerBackend,
)
from aiodoo_training.registries import (
    adaptation_registry,
    callback_registry,
    checkpoint_store_registry,
    curriculum_registry,
    dataset_source_registry,
    distributed_backend_registry,
    distributed_sampler_registry,
    evaluator_registry,
    exporter_registry,
    model_backend_registry,
    optimizer_registry,
    packing_registry,
    placement_strategy_registry,
    resource_planner_registry,
    rng_registry,
    sampling_registry,
    scheduler_registry,
    tokenizer_registry,
    tracker_registry,
    trainer_registry,
)
from aiodoo_training.registries.base import Registry


def _require_key[T](registry: Registry[T], key: str, kind: str) -> None:
    if registry.exists(key):
        return
    known = ", ".join(registry.list()) or "(none)"
    raise FactoryError(f"No {kind} registered for key '{key}'. Known keys: {known}.")


class DatasetSourceFactory:
    """Construct :class:`DatasetSource` implementations from the registry."""

    def __init__(self, registry: Registry[type[DatasetSource]] | None = None) -> None:
        # Prefer ``is None`` — empty Registry is falsy via ``__len__``.
        self._registry = dataset_source_registry if registry is None else registry

    def create(self, key: str) -> DatasetSource:
        _require_key(self._registry, key, "DatasetSource")
        source_cls = self._registry.get(key)
        return source_cls()


class TokenizerFactory:
    """Construct :class:`TokenizerPort` implementations from the registry."""

    def __init__(self, registry: Registry[type[TokenizerPort]] | None = None) -> None:
        self._registry = tokenizer_registry if registry is None else registry

    def create(self, key: str) -> TokenizerPort:
        _require_key(self._registry, key, "TokenizerPort")
        tokenizer_cls = self._registry.get(key)
        return tokenizer_cls()


class ModelBackendFactory:
    """Construct :class:`ModelBackend` implementations from the registry."""

    def __init__(self, registry: Registry[type[ModelBackend]] | None = None) -> None:
        self._registry = model_backend_registry if registry is None else registry

    def create(self, key: str = "stub") -> ModelBackend:
        _require_key(self._registry, key, "ModelBackend")
        backend_cls = self._registry.get(key)
        return backend_cls()


class AdaptationStrategyFactory:
    """Construct :class:`AdaptationStrategy` implementations from the registry."""

    def __init__(self, registry: Registry[type[AdaptationStrategy]] | None = None) -> None:
        self._registry = adaptation_registry if registry is None else registry

    def create(self, key: str = "lora") -> AdaptationStrategy:
        _require_key(self._registry, key, "AdaptationStrategy")
        strategy_cls = self._registry.get(key)
        return strategy_cls()


class TrainerBackendFactory:
    """Construct :class:`TrainerBackend` implementations from the registry."""

    def __init__(self, registry: Registry[type[TrainerBackend]] | None = None) -> None:
        self._registry = trainer_registry if registry is None else registry

    def create(self, key: str = "stub") -> TrainerBackend:
        _require_key(self._registry, key, "TrainerBackend")
        backend_cls = self._registry.get(key)
        return backend_cls()


class CheckpointStoreFactory:
    """Construct :class:`CheckpointStore` implementations from the registry."""

    def __init__(self, registry: Registry[type[CheckpointStore]] | None = None) -> None:
        self._registry = checkpoint_store_registry if registry is None else registry

    def create(self, key: str = "stub") -> CheckpointStore:
        _require_key(self._registry, key, "CheckpointStore")
        store_cls = self._registry.get(key)
        return store_cls()


class EvaluatorFactory:
    """Construct :class:`Evaluator` implementations from the registry."""

    def __init__(self, registry: Registry[type[Evaluator]] | None = None) -> None:
        self._registry = evaluator_registry if registry is None else registry

    def create(self, key: str = "stub") -> Evaluator:
        _require_key(self._registry, key, "Evaluator")
        evaluator_cls = self._registry.get(key)
        return evaluator_cls()


class ExporterFactory:
    """Construct :class:`Exporter` implementations from the registry."""

    def __init__(self, registry: Registry[type[Exporter]] | None = None) -> None:
        self._registry = exporter_registry if registry is None else registry

    def create(self, key: str = "stub") -> Exporter:
        _require_key(self._registry, key, "Exporter")
        exporter_cls = self._registry.get(key)
        return exporter_cls()


class TrackerFactory:
    """Construct :class:`ExperimentTracker` implementations from the registry."""

    def __init__(self, registry: Registry[type[ExperimentTracker]] | None = None) -> None:
        self._registry = tracker_registry if registry is None else registry

    def create(self, key: str = "null") -> ExperimentTracker:
        _require_key(self._registry, key, "ExperimentTracker")
        return self._registry.get(key)()


class ResourcePlannerFactory:
    """Construct :class:`ResourcePlanner` implementations from the registry."""

    def __init__(self, registry: Registry[type[ResourcePlanner]] | None = None) -> None:
        self._registry = resource_planner_registry if registry is None else registry

    def create(self, key: str = "static") -> ResourcePlanner:
        _require_key(self._registry, key, "ResourcePlanner")
        planner_cls = self._registry.get(key)
        return planner_cls()


class RngControllerFactory:
    """Construct :class:`RngController` implementations from the registry."""

    def __init__(self, registry: Registry[type[RngController]] | None = None) -> None:
        self._registry = rng_registry if registry is None else registry

    def create(self, key: str = "python") -> RngController:
        _require_key(self._registry, key, "RngController")
        rng_cls = self._registry.get(key)
        return rng_cls()


class OptimizerBackendFactory:
    """Construct :class:`OptimizerBackend` implementations from the registry."""

    def __init__(self, registry: Registry[type[OptimizerBackend]] | None = None) -> None:
        self._registry = optimizer_registry if registry is None else registry

    def create(self, key: str = "adamw") -> OptimizerBackend:
        _require_key(self._registry, key, "OptimizerBackend")
        backend_cls = self._registry.get(key)
        return backend_cls()


class SchedulerBackendFactory:
    """Construct :class:`SchedulerBackend` implementations from the registry."""

    def __init__(self, registry: Registry[type[SchedulerBackend]] | None = None) -> None:
        self._registry = scheduler_registry if registry is None else registry

    def create(self, key: str = "cosine") -> SchedulerBackend:
        _require_key(self._registry, key, "SchedulerBackend")
        backend_cls = self._registry.get(key)
        return backend_cls()


class CallbackFactory:
    """Construct :class:`TrainingCallback` implementations from the registry."""

    def __init__(self, registry: Registry[type[TrainingCallback]] | None = None) -> None:
        self._registry = callback_registry if registry is None else registry

    def create(self, key: str = "null") -> TrainingCallback:
        _require_key(self._registry, key, "TrainingCallback")
        callback_cls = self._registry.get(key)
        return callback_cls()


class PackingStrategyFactory:
    """Construct :class:`PackingStrategy` implementations from the registry."""

    def __init__(self, registry: Registry[type[PackingStrategy]] | None = None) -> None:
        self._registry = packing_registry if registry is None else registry

    def create(self, key: str = "none") -> PackingStrategy:
        _require_key(self._registry, key, "PackingStrategy")
        return self._registry.get(key)()


class CurriculumStrategyFactory:
    """Construct :class:`CurriculumStrategy` implementations from the registry."""

    def __init__(self, registry: Registry[type[CurriculumStrategy]] | None = None) -> None:
        self._registry = curriculum_registry if registry is None else registry

    def create(self, key: str = "none") -> CurriculumStrategy:
        _require_key(self._registry, key, "CurriculumStrategy")
        return self._registry.get(key)()


class SamplingStrategyFactory:
    """Construct :class:`SamplingStrategy` implementations from the registry."""

    def __init__(self, registry: Registry[type[SamplingStrategy]] | None = None) -> None:
        self._registry = sampling_registry if registry is None else registry

    def create(self, key: str = "identity") -> SamplingStrategy:
        _require_key(self._registry, key, "SamplingStrategy")
        return self._registry.get(key)()


class DistributedBackendFactory:
    """Construct :class:`DistributedBackend` implementations from the registry."""

    def __init__(self, registry: Registry[type[DistributedBackend]] | None = None) -> None:
        self._registry = distributed_backend_registry if registry is None else registry

    def create(self, key: str = "fake") -> DistributedBackend:
        _require_key(self._registry, key, "DistributedBackend")
        return self._registry.get(key)()


class PlacementStrategyFactory:
    """Construct :class:`PlacementStrategy` implementations from the registry."""

    def __init__(self, registry: Registry[type[PlacementStrategy]] | None = None) -> None:
        self._registry = placement_strategy_registry if registry is None else registry

    def create(self, key: str = "single") -> PlacementStrategy:
        _require_key(self._registry, key, "PlacementStrategy")
        return self._registry.get(key)()


class DistributedSamplerFactory:
    """Construct :class:`DistributedSampler` implementations from the registry."""

    def __init__(self, registry: Registry[type[DistributedSampler]] | None = None) -> None:
        self._registry = distributed_sampler_registry if registry is None else registry

    def create(self, key: str = "shard") -> DistributedSampler:
        _require_key(self._registry, key, "DistributedSampler")
        return self._registry.get(key)()
