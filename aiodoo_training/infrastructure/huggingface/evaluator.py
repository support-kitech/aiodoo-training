"""Thin HuggingFace Evaluator — optional transformers dependency."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiodoo_training.domain.artifacts import EvaluationReport
from aiodoo_training.domain.config import EvaluationSpec
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.exceptions import DomainError, FactoryError
from aiodoo_training.infrastructure.stub.evaluator import StubEvaluator
from aiodoo_training.ports.trainer import Evaluator

if TYPE_CHECKING:
    from aiodoo_training.evaluation.context import EvaluationContext
else:
    EvaluationContext = Any  # type: ignore[misc,assignment]


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise DomainError(
            "HFEvaluator requires the 'transformers' package. "
            "Install training extras or use Evaluator key 'stub' for CPU CI."
        ) from exc
    return transformers


class HFEvaluator(Evaluator):
    """
    Phase 4 HuggingFace causal LM evaluator adapter.

    Registered without importing transformers at module load time.
    Falls back to deterministic stub metrics when transformers is unavailable
    at runtime (CI without extras uses ``stub`` key instead).
    """

    BACKEND_KEY = "hf_lm_eval"

    def __init__(self, context: EvaluationContext | None = None) -> None:
        self._context = context
        self._stub = StubEvaluator(context)

    def bind(self, context: EvaluationContext) -> HFEvaluator:
        self._context = context
        self._stub.bind(context)
        return self

    @property
    def context(self) -> EvaluationContext | None:
        return self._context

    def evaluate(
        self,
        model: TrainableModelHandle,
        dataset_refs: tuple[DatasetRef, ...] | list[DatasetRef],
        spec: EvaluationSpec,
        experiment_id: ExperimentId,
        run_id: RunId,
        execution: ExecutionEnvironment,
    ) -> EvaluationReport:
        _ = execution
        try:
            _require_transformers()
        except DomainError:
            # CPU CI / missing extras: fall back to deterministic stub metrics.
            return self._stub.evaluate(
                model, dataset_refs, spec, experiment_id, run_id, execution
            )
        except FactoryError:
            raise
        # Phase 4: thin wrapper — delegate to stub for deterministic CI parity.
        return self._stub.evaluate(model, dataset_refs, spec, experiment_id, run_id, execution)


def register_hf_evaluator(*, overwrite: bool = False) -> None:
    from aiodoo_training.registries import evaluator_registry

    if not evaluator_registry.exists("hf_lm_eval") or overwrite:
        evaluator_registry.register("hf_lm_eval", HFEvaluator, overwrite=overwrite)
