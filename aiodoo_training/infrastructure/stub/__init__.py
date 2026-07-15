"""CPU stub trainer package — no Torch / Transformers required."""

from aiodoo_training.infrastructure.stub.checkpoint_store import (
    StubCheckpointStore,
    register_default_checkpoint_stores,
)
from aiodoo_training.infrastructure.stub.evaluator import (
    StubEvaluator,
    register_default_evaluators,
)
from aiodoo_training.infrastructure.stub.exporter import (
    StubExporter,
    register_default_exporters,
)
from aiodoo_training.infrastructure.stub.trainer import (
    StubTrainerBackend,
    register_default_trainers,
)

__all__ = [
    "StubCheckpointStore",
    "StubEvaluator",
    "StubExporter",
    "StubTrainerBackend",
    "register_default_checkpoint_stores",
    "register_default_evaluators",
    "register_default_exporters",
    "register_default_trainers",
]
