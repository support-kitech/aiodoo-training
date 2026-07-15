"""Tokenizer port."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from aiodoo_training.domain.examples import TokenBatch, TrainingExample
from aiodoo_training.domain.refs import ModelRef


class TokenizerPort(ABC):
    """
    Family-agnostic tokenizer.

    Chat formatting is resolved via ChatTemplateRegistry — implementations must
    not hardcode model-family prompt layouts.
    """

    @abstractmethod
    def load(self, model_ref: ModelRef) -> None:
        """Load tokenizer assets associated with the model reference."""

    @abstractmethod
    def encode_examples(self, examples: Sequence[TrainingExample]) -> TokenBatch:
        """Tokenize a batch of training examples into a TokenBatch."""

    @abstractmethod
    def vocabulary_size(self) -> int:
        """Return the tokenizer vocabulary size."""

    def fingerprint(self) -> str:
        """Optional tokenizer identity fingerprint (override in backends)."""
        return f"{type(self).__name__}:vocab={self.vocabulary_size()}"
