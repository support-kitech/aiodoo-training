"""Thin HuggingFace TrainerBackend — optional transformers dependency."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.domain.training import CheckpointHandle, TrainingProgress
from aiodoo_training.exceptions import DomainError, FactoryError
from aiodoo_training.ports.trainer import TrainerBackend

if TYPE_CHECKING:
    from aiodoo_training.domain.config import ExperimentConfig
    from aiodoo_training.training.context import TrainingContext
else:
    ExperimentConfig = Any  # type: ignore[misc,assignment]
    TrainingContext = Any  # type: ignore[misc,assignment]


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise FactoryError(
            "HFTrainerBackend requires the 'transformers' package. "
            "Install training extras or use TrainerBackend key 'stub' for CPU CI."
        ) from exc
    return transformers


class HFTrainerBackend(TrainerBackend):
    """
    Phase 3 HuggingFace Trainer adapter.

    Registered and constructible even without transformers installed; ``train`` /
    ``resume`` fail with a clear :class:`FactoryError` / :class:`DomainError`
    rather than importing transformers at module load time.
    """

    BACKEND_KEY = "hf_trainer"

    def __init__(self, context: TrainingContext | None = None) -> None:
        self._context = context

    def bind(self, context: TrainingContext) -> HFTrainerBackend:
        self._context = context
        return self

    @property
    def context(self) -> TrainingContext | None:
        return self._context

    def train(
        self,
        config: ExperimentConfig,
        model: TrainableModelHandle,
        execution: ExecutionEnvironment,
    ) -> TrainingProgress:
        _ = (config, execution)
        transformers = _require_transformers()
        return self._run_or_reject(model, transformers, resume=False)

    def resume(
        self,
        config: ExperimentConfig,
        model: TrainableModelHandle,
        checkpoint: CheckpointHandle,
        execution: ExecutionEnvironment,
    ) -> TrainingProgress:
        _ = (config, checkpoint, execution)
        transformers = _require_transformers()
        return self._run_or_reject(model, transformers, resume=True)

    def _run_or_reject(
        self,
        model: TrainableModelHandle,
        transformers: Any,
        *,
        resume: bool,
    ) -> TrainingProgress:
        _ = resume
        from aiodoo_training.infrastructure.model_handles import require_trainable_carrier

        carrier = require_trainable_carrier(model)
        framework = carrier.framework_model
        if isinstance(framework, dict) and framework.get("kind") == "stub":
            raise DomainError(
                "HFTrainerBackend cannot train stub framework models. "
                "Use trainer backend key 'stub' for CPU CI."
            )
        # Real HF wiring is intentionally thin in Phase 3 — actual train loops
        # require model extras / GPU stacks beyond CI defaults.
        _ = transformers
        raise DomainError(
            "HFTrainerBackend is registered but Phase 3 CI uses StubTrainerBackend. "
            "Full HuggingFace Trainer execution requires a loaded Transformers model "
            "and training extras (GPU/optional deps). Install extras and bind a "
            "non-stub TrainableModelHandle to enable."
        )


def register_hf_trainer(*, overwrite: bool = False) -> None:
    """Register the HF trainer under ``hf_trainer`` (lazy — no import of transformers)."""
    from aiodoo_training.registries import trainer_registry

    if not trainer_registry.exists("hf_trainer") or overwrite:
        trainer_registry.register("hf_trainer", HFTrainerBackend, overwrite=overwrite)
