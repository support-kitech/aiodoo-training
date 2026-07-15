"""CPU stub Evaluator — deterministic metrics for CI golden tests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aiodoo_training.domain.artifacts import EvaluationReport
from aiodoo_training.domain.config import EvaluationSpec
from aiodoo_training.domain.enums import EvaluationStatus
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.domain.training import MetricSnapshot
from aiodoo_training.infrastructure.model_handles import require_trainable_carrier
from aiodoo_training.infrastructure.stub.trainer import stub_loss
from aiodoo_training.ports.trainer import Evaluator

if TYPE_CHECKING:
    from aiodoo_training.evaluation.context import EvaluationContext
else:
    EvaluationContext = Any  # type: ignore[misc,assignment]


def _hash_mod(*parts: str) -> int:
    material = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16) % 1000


def stub_eval_loss(*, model_fp: str, dataset_key: str, seed: int) -> float:
    """Deterministic scalar loss from portable identifiers."""
    hash_mod = _hash_mod(model_fp, dataset_key)
    return 1.0 / (1.0 + hash_mod + seed * 1e-6)


def stub_eval_perplexity(loss: float) -> float:
    """Deterministic perplexity derived from loss."""
    import math

    return math.exp(min(loss, 20.0))


def stub_eval_token_accuracy(*, loss: float, seed: int) -> float:
    """Deterministic token accuracy in (0, 1] derived from loss + seed."""
    raw = 1.0 - loss / (1.0 + loss) + seed * 1e-8
    return max(0.0, min(1.0, raw))


class StubEvaluator(Evaluator):
    """
    Deterministic CPU evaluator for CI golden tests.

    Rich session collaborators arrive via :meth:`bind` — frozen
    ``evaluate`` signature is unchanged. Never exports artifacts.
    """

    BACKEND_KEY = "stub"

    def __init__(self, context: EvaluationContext | None = None) -> None:
        self._context = context
        self._lifecycle = None
        try:
            from aiodoo_training.evaluation.lifecycle import EvaluationLifecycle

            self._lifecycle = EvaluationLifecycle()
        except ImportError:
            self._lifecycle = None

    def bind(self, context: EvaluationContext) -> StubEvaluator:
        """Attach a resolved :class:`EvaluationContext` without widening the port."""
        self._context = context
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
        carrier = require_trainable_carrier(model)
        fw = carrier.framework_model
        weight_sum = 0.0
        if isinstance(fw, dict):
            weights = fw.get("weights")
            if isinstance(weights, (list, tuple)):
                weight_sum = float(sum(float(w) for w in weights))

        ctx = self._context
        seed = 42
        if ctx is not None and hasattr(ctx, "evaluation_policy"):
            policy_seed = ctx.evaluation_policy.seed
            if policy_seed is None and hasattr(ctx, "config"):
                policy_seed = ctx.config.seed
            if policy_seed is not None:
                seed = int(policy_seed)

        model_fp = ""
        if ctx is not None:
            model_fp = ctx.model_fingerprint or ""
        if not model_fp and isinstance(fw, dict):
            model_fp = str(fw.get("fingerprint", weight_sum))

        session = self._resolve_session(experiment_id, run_id)
        session = self._transition_start(session)

        refs = tuple(dataset_refs)
        dataset_key = "|".join(sorted(str(ref.path) for ref in refs)) or "no-dataset"
        base_loss = stub_eval_loss(
            model_fp=model_fp or str(weight_sum),
            dataset_key=dataset_key,
            seed=seed,
        )
        # Also incorporate weight_sum like stub trainer for model-state sensitivity.
        loss = stub_loss(step=0, weight_sum=weight_sum, seed=seed) * 0.5 + base_loss * 0.5
        perplexity = stub_eval_perplexity(loss)
        token_accuracy = stub_eval_token_accuracy(loss=loss, seed=seed)

        now = datetime.now(UTC)
        metrics = (
            MetricSnapshot(name="loss", value=loss, step=1, timestamp=now),
            MetricSnapshot(name="perplexity", value=perplexity, step=1, timestamp=now),
            MetricSnapshot(name="token_accuracy", value=token_accuracy, step=1, timestamp=now),
        )

        session = self._transition_complete(session)
        self._sync_session(session)

        return EvaluationReport(
            experiment_id=experiment_id,
            run_id=run_id,
            metrics=metrics,
            passed=True,
            details="stub evaluator",
            created_at=now,
        )

    def _resolve_session(self, experiment_id: ExperimentId, run_id: RunId):
        ctx = self._context
        if ctx is not None and getattr(ctx, "evaluation_session", None) is not None:
            return ctx.evaluation_session
        from aiodoo_training.domain.evaluation_session import EvaluationSession

        return EvaluationSession(
            session_id="stub-eval-session",
            experiment_id=experiment_id,
            run_id=run_id,
        )

    def _transition_start(self, session):
        lifecycle = self._lifecycle
        if lifecycle is None:
            return session.with_status(EvaluationStatus.RUNNING)
        if session.status == EvaluationStatus.PENDING:
            return lifecycle.start(session)
        return session

    def _transition_complete(self, session):
        lifecycle = self._lifecycle
        if lifecycle is None:
            return session.with_status(EvaluationStatus.COMPLETED)
        if session.status == EvaluationStatus.RUNNING:
            return lifecycle.complete(session)
        return session.with_status(EvaluationStatus.COMPLETED)

    def _sync_session(self, session) -> None:
        ctx = self._context
        if ctx is None or not hasattr(ctx, "with_evaluation_session"):
            return
        self._context = ctx.with_evaluation_session(session)


def register_default_evaluators(*, overwrite: bool = False) -> None:
    """Register ``stub`` and lazy ``hf_lm_eval`` evaluators."""
    from aiodoo_training.infrastructure.huggingface.evaluator import register_hf_evaluator
    from aiodoo_training.registries import evaluator_registry

    if not evaluator_registry.exists("stub") or overwrite:
        evaluator_registry.register("stub", StubEvaluator, overwrite=overwrite)
    register_hf_evaluator(overwrite=overwrite)
