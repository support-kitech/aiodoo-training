"""Factories that construct port implementations from registries."""

from aiodoo_training.factories.factories import (
    AdaptationStrategyFactory,
    CallbackFactory,
    CheckpointStoreFactory,
    CurriculumStrategyFactory,
    DatasetSourceFactory,
    DistributedBackendFactory,
    DistributedSamplerFactory,
    EvaluatorFactory,
    ExporterFactory,
    ModelBackendFactory,
    OptimizerBackendFactory,
    PackingStrategyFactory,
    PlacementStrategyFactory,
    ResourcePlannerFactory,
    RngControllerFactory,
    SamplingStrategyFactory,
    SchedulerBackendFactory,
    TokenizerFactory,
    TrackerFactory,
    TrainerBackendFactory,
)

# Aliases matching Phase 4 architecture naming.
EvaluationFactory = EvaluatorFactory
ExportFactory = ExporterFactory

__all__ = [
    "AdaptationStrategyFactory",
    "CallbackFactory",
    "CheckpointStoreFactory",
    "CurriculumStrategyFactory",
    "DatasetSourceFactory",
    "DistributedBackendFactory",
    "DistributedSamplerFactory",
    "EvaluationFactory",
    "EvaluatorFactory",
    "ExportFactory",
    "ExporterFactory",
    "ModelBackendFactory",
    "OptimizerBackendFactory",
    "PackingStrategyFactory",
    "PlacementStrategyFactory",
    "ResourcePlannerFactory",
    "RngControllerFactory",
    "SamplingStrategyFactory",
    "SchedulerBackendFactory",
    "TokenizerFactory",
    "TrackerFactory",
    "TrainerBackendFactory",
]
