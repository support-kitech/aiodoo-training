"""Packing, curriculum, and sampling ports."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from aiodoo_training.domain.config import CurriculumSpec, PackingSpec
from aiodoo_training.domain.examples import TokenBatch, TrainingExample
from aiodoo_training.domain.packing_policies import SamplingSpec


class PackingStrategy(ABC):
    """Packs tokenized or raw examples into efficient training batches."""

    @abstractmethod
    def pack(
        self,
        examples: Sequence[TrainingExample],
        spec: PackingSpec,
    ) -> Iterator[TokenBatch]:
        """Yield packed batches according to the packing specification."""


class CurriculumStrategy(ABC):
    """Orders or weights training stages for curriculum learning."""

    @abstractmethod
    def plan(
        self,
        examples: Sequence[TrainingExample],
        spec: CurriculumSpec,
    ) -> Sequence[Sequence[TrainingExample]]:
        """
        Return ordered curriculum stages.

        Each inner sequence is one training stage worth of examples.
        """


class SamplingStrategy(ABC):
    """Additive Phase 5 port: reorder or subsample examples deterministically."""

    @abstractmethod
    def sample(
        self,
        examples: Sequence[TrainingExample],
        spec: SamplingSpec,
    ) -> Sequence[TrainingExample]:
        """Return a deterministically ordered example sequence."""
