"""Phase 4 EvaluationBuilder / EvaluationContextBuilder."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from aiodoo_training.domain.config import EvaluationSpec, ExperimentConfig
from aiodoo_training.domain.evaluation_policies import AcceptancePolicy, EvaluationPolicy
from aiodoo_training.domain.evaluation_session import EvaluationSession
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.exceptions import BuilderError

if TYPE_CHECKING:
    from aiodoo_training.evaluation.context import EvaluationContext
from aiodoo_training.ports.trainer import Evaluator, ExperimentTracker, RngController


class EvaluationBuilder:
    """Assembles EvaluationPolicy / AcceptancePolicy from profiles and fragments."""

    def __init__(self) -> None:
        self._backend_key = "stub"
        self._profile_key = "default"
        self._metrics: tuple[str, ...] = ("loss", "perplexity", "token_accuracy")
        self._max_examples: int | None = None
        self._seed: int | None = 42
        self._acceptance = AcceptancePolicy()

    def with_backend(self, key: str) -> EvaluationBuilder:
        self._backend_key = key
        return self

    def with_profile(self, key: str) -> EvaluationBuilder:
        self._profile_key = key
        return self

    def with_metrics(self, *metrics: str) -> EvaluationBuilder:
        self._metrics = tuple(metrics)
        return self

    def with_seed(self, seed: int | None) -> EvaluationBuilder:
        self._seed = seed
        return self

    def with_acceptance(self, policy: AcceptancePolicy) -> EvaluationBuilder:
        self._acceptance = policy
        return self

    def build_policy(self) -> EvaluationPolicy:
        return EvaluationPolicy(
            backend_key=self._backend_key,
            profile_key=self._profile_key,
            metrics=self._metrics,
            max_examples=self._max_examples,
            seed=self._seed,
        )

    def build_acceptance(self) -> AcceptancePolicy:
        return self._acceptance


class EvaluationContextBuilder:
    """Builds a resolved :class:`EvaluationContext` from collaborator pieces."""

    def __init__(self) -> None:
        self._pieces: dict[str, object] = {}

    def with_config(self, config: ExperimentConfig) -> EvaluationContextBuilder:
        self._pieces["config"] = config
        return self

    def with_piece(self, key: str, value: object) -> EvaluationContextBuilder:
        self._pieces[key] = value
        return self

    def build(self, config: ExperimentConfig | None = None) -> EvaluationContext:
        cfg = config if config is not None else self._pieces.get("config")
        if not isinstance(cfg, ExperimentConfig):
            raise BuilderError("EvaluationContextBuilder requires ExperimentConfig.")

        required = ("execution", "model", "evaluator", "evaluation_session")
        missing = [key for key in required if key not in self._pieces]
        if missing:
            raise BuilderError(
                "EvaluationContextBuilder missing required pieces: " + ", ".join(missing)
            )

        evaluation_spec = self._pieces.get("evaluation_spec")
        if not isinstance(evaluation_spec, EvaluationSpec):
            evaluation_spec = cfg.evaluation

        evaluation_policy = self._pieces.get("evaluation_policy")
        if not isinstance(evaluation_policy, EvaluationPolicy):
            evaluation_policy = EvaluationPolicy(
                backend_key=str(self._pieces.get("evaluator_backend_key") or "stub"),
                seed=cfg.seed,
            )

        acceptance_policy = self._pieces.get("acceptance_policy")
        if not isinstance(acceptance_policy, AcceptancePolicy):
            acceptance_policy = AcceptancePolicy()

        dataset_refs_obj = self._pieces.get("dataset_refs")
        if dataset_refs_obj is None:
            dataset_refs = evaluation_spec.dataset_refs
        elif isinstance(dataset_refs_obj, tuple):
            dataset_refs = cast(tuple[DatasetRef, ...], dataset_refs_obj)
        else:
            dataset_refs = cast(tuple[DatasetRef, ...], tuple(dataset_refs_obj))  # type: ignore[arg-type]

        sessions_obj = self._pieces.get("dataset_sessions") or ()
        dataset_sessions = cast(tuple[DatasetSession, ...], tuple(sessions_obj))  # type: ignore[arg-type]

        bind_extra_obj = self._pieces.get("bind_extra") or {}
        if not isinstance(bind_extra_obj, dict):
            raise BuilderError("bind_extra must be a mapping when provided.")
        bind_extra: dict[str, Any] = {str(k): v for k, v in bind_extra_obj.items()}

        from aiodoo_training.evaluation.context import EvaluationContext

        return EvaluationContext(
            config=cfg,
            execution=self._pieces["execution"],  # type: ignore[arg-type]
            model=self._pieces["model"],  # type: ignore[arg-type]
            evaluation_spec=evaluation_spec,
            evaluation_policy=evaluation_policy,
            acceptance_policy=acceptance_policy,
            evaluation_session=self._pieces["evaluation_session"],  # type: ignore[arg-type]
            dataset_refs=dataset_refs,
            dataset_sessions=dataset_sessions,
            evaluator=self._pieces["evaluator"],  # type: ignore[arg-type]
            evaluator_backend_key=str(self._pieces.get("evaluator_backend_key") or "stub"),
            model_fingerprint=str(self._pieces.get("model_fingerprint") or ""),
            adapter_fingerprint=str(self._pieces.get("adapter_fingerprint") or ""),
            config_fingerprint=str(self._pieces.get("config_fingerprint") or ""),
            execution_digest=str(self._pieces.get("execution_digest") or ""),
            rng=self._pieces.get("rng"),  # type: ignore[arg-type]
            tracker=self._pieces.get("tracker"),  # type: ignore[arg-type]
            evaluation_report=self._pieces.get("evaluation_report"),  # type: ignore[arg-type]
            bind_extra=bind_extra,
        )


def make_evaluation_session(
    *,
    experiment_id: ExperimentId,
    run_id: RunId,
    model_fingerprint: str = "",
    adapter_fingerprint: str = "",
    config_fingerprint: str = "",
    execution_digest: str = "",
) -> EvaluationSession:
    """Create a PENDING EvaluationSession with stable-looking identity."""
    now = datetime.now(UTC)
    return EvaluationSession(
        session_id=f"eval-{uuid4().hex[:12]}",
        experiment_id=experiment_id,
        run_id=run_id,
        model_fingerprint=model_fingerprint,
        adapter_fingerprint=adapter_fingerprint,
        config_fingerprint=config_fingerprint,
        execution_digest=execution_digest,
        created_at=now,
        updated_at=now,
    )


def enable_evaluation(
    config: ExperimentConfig,
    *,
    dataset_refs: tuple[DatasetRef, ...] | None = None,
) -> ExperimentConfig:
    """Return a copy of ExperimentConfig with evaluation enabled."""
    refs = dataset_refs if dataset_refs is not None else config.evaluation.dataset_refs
    return replace(
        config,
        evaluation=replace(config.evaluation, enabled=True, dataset_refs=refs),
    )


# silence unused-import noise for documented builder collaborators
_ = (TrainableModelHandle, ExecutionEnvironment, Evaluator, ExperimentTracker, RngController)
