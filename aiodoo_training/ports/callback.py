"""Training callback port — synchronous event listener."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiodoo_training.domain.training_events import TrainingEvent
    from aiodoo_training.training.context import CallbackContext


class TrainingCallback(ABC):
    """Synchronous training event listener."""

    @abstractmethod
    def on_event(self, event: TrainingEvent, context: CallbackContext) -> None:
        """Handle a training event in registration order."""
