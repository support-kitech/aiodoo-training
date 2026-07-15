"""Named registry catalog (empty until defaults are registered)."""

from typing import Any

from aiodoo_training.domain.adapter_info import AdapterProfile
from aiodoo_training.domain.enums import ModelFamily
from aiodoo_training.domain.evaluation_policies import EvaluationProfile, ExportProfile
from aiodoo_training.domain.metric_definition import MetricDefinition
from aiodoo_training.domain.model_info import ModelCapabilities, ModelProfile
from aiodoo_training.ports.callback import TrainingCallback
from aiodoo_training.ports.chat_template import ChatTemplate
from aiodoo_training.ports.dataset import DatasetSource, ExampleFormatter
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
from aiodoo_training.registries.base import Registry

formatter_registry: Registry[type[ExampleFormatter]] = Registry("formatters")
dataset_source_registry: Registry[type[DatasetSource]] = Registry("dataset_sources")
tokenizer_registry: Registry[type[TokenizerPort]] = Registry("tokenizers")
chat_template_registry: Registry[type[ChatTemplate]] = Registry("chat_templates")
model_backend_registry: Registry[type[ModelBackend]] = Registry("model_backends")
adaptation_registry: Registry[type[AdaptationStrategy]] = Registry("adaptation_strategies")
# Declarative adapter metadata (independent of AdaptationStrategy implementations).
adapter_registry: Registry[AdapterProfile] = Registry("adapter_profiles")
packing_registry: Registry[type[PackingStrategy]] = Registry("packing_strategies")
curriculum_registry: Registry[type[CurriculumStrategy]] = Registry("curriculum_strategies")
sampling_registry: Registry[type[SamplingStrategy]] = Registry("sampling_strategies")
trainer_registry: Registry[type[TrainerBackend]] = Registry("trainers")
checkpoint_store_registry: Registry[type[CheckpointStore]] = Registry("checkpoint_stores")
evaluator_registry: Registry[type[Evaluator]] = Registry("evaluators")
exporter_registry: Registry[type[Exporter]] = Registry("exporters")
tracker_registry: Registry[type[ExperimentTracker]] = Registry("trackers")
resource_planner_registry: Registry[type[ResourcePlanner]] = Registry("resource_planners")
# Phase 2 — model catalogs (registration-driven; reuse Registry[T])
model_family_registry: Registry[ModelFamily] = Registry("model_families")
model_profile_registry: Registry[ModelProfile] = Registry("model_profiles")
model_capability_registry: Registry[ModelCapabilities] = Registry("model_capabilities")
# Phase 3 — training infrastructure
optimizer_registry: Registry[type[OptimizerBackend]] = Registry("optimizers")
scheduler_registry: Registry[type[SchedulerBackend]] = Registry("schedulers")
rng_registry: Registry[type[RngController]] = Registry("rng_controllers")
callback_registry: Registry[type[TrainingCallback]] = Registry("callbacks")
# Phase 4 — evaluation / export catalogs
metric_registry: Registry[MetricDefinition] = Registry("metrics")
evaluation_profile_registry: Registry[EvaluationProfile] = Registry("evaluation_profiles")
export_profile_registry: Registry[ExportProfile] = Registry("export_profiles")
# Phase 7 — distributed readiness
distributed_backend_registry: Registry[type[DistributedBackend]] = Registry(
    "distributed_backends"
)
placement_strategy_registry: Registry[type[PlacementStrategy]] = Registry(
    "placement_strategies"
)
distributed_sampler_registry: Registry[type[DistributedSampler]] = Registry(
    "distributed_samplers"
)


def all_registries() -> tuple[Registry[Any], ...]:
    """Return all module-level registries for inspection and tests."""
    return (
        formatter_registry,
        dataset_source_registry,
        tokenizer_registry,
        chat_template_registry,
        model_backend_registry,
        adaptation_registry,
        adapter_registry,
        packing_registry,
        curriculum_registry,
        sampling_registry,
        trainer_registry,
        checkpoint_store_registry,
        evaluator_registry,
        exporter_registry,
        tracker_registry,
        resource_planner_registry,
        model_family_registry,
        model_profile_registry,
        model_capability_registry,
        optimizer_registry,
        scheduler_registry,
        rng_registry,
        callback_registry,
        metric_registry,
        evaluation_profile_registry,
        export_profile_registry,
        distributed_backend_registry,
        placement_strategy_registry,
        distributed_sampler_registry,
    )


def clear_all_registries() -> None:
    """Clear every registry (tests only; raises if any registry is frozen)."""
    for registry in all_registries():
        if not registry.is_frozen:
            registry.clear()


__all__ = [
    "adaptation_registry",
    "adapter_registry",
    "all_registries",
    "callback_registry",
    "chat_template_registry",
    "checkpoint_store_registry",
    "clear_all_registries",
    "curriculum_registry",
    "dataset_source_registry",
    "distributed_backend_registry",
    "distributed_sampler_registry",
    "evaluation_profile_registry",
    "evaluator_registry",
    "export_profile_registry",
    "exporter_registry",
    "formatter_registry",
    "metric_registry",
    "model_backend_registry",
    "model_capability_registry",
    "model_family_registry",
    "model_profile_registry",
    "optimizer_registry",
    "packing_registry",
    "placement_strategy_registry",
    "resource_planner_registry",
    "rng_registry",
    "sampling_registry",
    "scheduler_registry",
    "tokenizer_registry",
    "tracker_registry",
    "trainer_registry",
]
