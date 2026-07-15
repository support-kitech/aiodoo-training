"""Evaluation runtime context — resolved application bag for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from aiodoo_training.domain.artifacts import EvaluationReport
from aiodoo_training.domain.config import EvaluationSpec, ExperimentConfig
from aiodoo_training.domain.evaluation_policies import AcceptancePolicy, EvaluationPolicy
from aiodoo_training.domain.evaluation_session import EvaluationSession
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.ports.trainer import Evaluator, ExperimentTracker, RngController


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """
    Resolved runtime collaborators for evaluation pipeline stages.

    Built by builders; consumed by stages and bindable for ``Evaluator``
    infrastructure adapters — never widens frozen port signatures.
    """

    config: ExperimentConfig
    execution: ExecutionEnvironment
    model: TrainableModelHandle
    evaluation_spec: EvaluationSpec
    evaluation_policy: EvaluationPolicy
    acceptance_policy: AcceptancePolicy
    evaluation_session: EvaluationSession
    dataset_refs: tuple[DatasetRef, ...]
    dataset_sessions: tuple[DatasetSession, ...]
    evaluator: Evaluator
    evaluator_backend_key: str = "stub"
    model_fingerprint: str = ""
    adapter_fingerprint: str = ""
    config_fingerprint: str = ""
    execution_digest: str = ""
    rng: RngController | None = None
    tracker: ExperimentTracker | None = None
    evaluation_report: EvaluationReport | None = None
    bind_extra: dict[str, Any] = field(default_factory=dict)

    def with_evaluation_session(self, session: EvaluationSession) -> EvaluationContext:
        return replace(self, evaluation_session=session)

    def with_evaluation_report(self, report: EvaluationReport) -> EvaluationContext:
        return replace(self, evaluation_report=report)

    def with_dataset_sessions(self, sessions: tuple[DatasetSession, ...]) -> EvaluationContext:
        return replace(self, dataset_sessions=sessions)
