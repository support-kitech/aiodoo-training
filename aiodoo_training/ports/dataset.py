"""Dataset-related ports."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from aiodoo_training.domain.config import DatasetMixSpec
from aiodoo_training.domain.examples import TrainingExample
from aiodoo_training.domain.refs import DatasetRef


class DatasetSource(ABC):
    """
    Loads protocol datasets referenced by DatasetRef values.

    Implementations belong in infrastructure (Phase 1+). Phase 0 defines the port only.
    """

    @abstractmethod
    def load(self, refs: Sequence[DatasetRef]) -> Iterator[TrainingExample]:
        """Stream normalized training examples from the given dataset refs."""

    @abstractmethod
    def load_mix(self, mix: DatasetMixSpec) -> Iterator[TrainingExample]:
        """Stream examples according to a weighted dataset mix specification."""


class ExampleFormatter(ABC):
    """
    Maps a raw protocol record (dict) into a domain TrainingExample.

    One formatter is registered per DatasetType / generator family.
    """

    @abstractmethod
    def supports(self, dataset_type: str) -> bool:
        """Return True if this formatter handles the given dataset type key."""

    @abstractmethod
    def format(self, record: dict[str, object], dataset_type: str) -> TrainingExample:
        """Convert a single protocol JSON object into a TrainingExample."""
