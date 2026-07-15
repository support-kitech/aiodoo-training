"""Chat template port — keeps model-family prompting out of tokenizers."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence


class ChatTemplate(ABC):
    """
    Formats role-tagged messages into a single prompt string for tokenization.

    Implementations are registered in ``chat_template_registry`` by family name
    (e.g. ``qwen``, ``llama``). Tokenizers resolve templates through the registry
    and must not hardcode family-specific formatting.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable registry key / template identity (e.g. ``qwen``)."""

    @property
    @abstractmethod
    def family(self) -> str:
        """Model family this template targets (may equal ``name``)."""

    @abstractmethod
    def render(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Render chat messages into a single deterministic text string."""

    @abstractmethod
    def fingerprint(self) -> str:
        """Canonical fingerprint of template identity and version."""
