"""Model and adaptation ports — framework-independent opaque handles only."""

from abc import ABC, abstractmethod

from aiodoo_training.domain.config import AdaptationSpec
from aiodoo_training.domain.handles import BaseModelHandle, TrainableModelHandle
from aiodoo_training.domain.refs import ModelRef
from aiodoo_training.domain.resources import ExecutionEnvironment


class ModelBackend(ABC):
    """
    Loads a base causal language model according to ModelRef + ExecutionEnvironment.

    Returns an opaque :class:`BaseModelHandle`. Never exposes Torch / HF types
    in this signature. Adaptation (LoRA/QLoRA/full) is a separate port.
    """

    @abstractmethod
    def load(
        self,
        model_ref: ModelRef,
        execution: ExecutionEnvironment,
    ) -> BaseModelHandle:
        """Load the base model under the resolved execution environment."""


class AdaptationStrategy(ABC):
    """
    Applies a parameter-efficient or full fine-tuning strategy to a base model.

    Must not assume PEFT, bitsandbytes, or any concrete library — those belong
    exclusively in infrastructure adapters.
    """

    @abstractmethod
    def apply(
        self,
        base_model: BaseModelHandle,
        spec: AdaptationSpec,
        execution: ExecutionEnvironment,
    ) -> TrainableModelHandle:
        """Return a trainable model handle derived from ``base_model``."""

    @abstractmethod
    def trainable_parameter_count(self, model: TrainableModelHandle) -> int:
        """Return the number of trainable parameters after adaptation."""
