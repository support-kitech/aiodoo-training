"""Phase 4 evaluation application layer."""

from aiodoo_training.evaluation.context import EvaluationContext
from aiodoo_training.evaluation.engine import EvaluationEngine
from aiodoo_training.evaluation.lifecycle import EvaluationLifecycle
from aiodoo_training.evaluation.metrics import MetricAggregator, MetricCollector, MetricHistory
from aiodoo_training.evaluation.quality_gate import ModelValidator, QualityGate

__all__ = [
    "EvaluationContext",
    "EvaluationEngine",
    "EvaluationLifecycle",
    "MetricAggregator",
    "MetricCollector",
    "MetricHistory",
    "ModelValidator",
    "QualityGate",
    "build_stub_evaluation_context",
    "run_stub_evaluate",
]


def __getattr__(name: str) -> object:
    if name == "build_stub_evaluation_context":
        from aiodoo_training.evaluation.harness import build_stub_evaluation_context

        return build_stub_evaluation_context
    if name == "run_stub_evaluate":
        from aiodoo_training.evaluation.harness import run_stub_evaluate

        return run_stub_evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
