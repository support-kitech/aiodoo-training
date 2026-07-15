"""Logging training callback — stderr / stdlib logging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiodoo_training.ports.callback import TrainingCallback

if TYPE_CHECKING:
    from aiodoo_training.domain.training_events import TrainingEvent
    from aiodoo_training.training.context import CallbackContext

_LOG = logging.getLogger("aiodoo_training.training")


class LoggingCallback(TrainingCallback):
    """Emit training events to the ``aiodoo_training.training`` logger."""

    def on_event(self, event: TrainingEvent, context: CallbackContext) -> None:
        _ = context
        parts = [
            f"event={event.kind.value}",
            f"step={event.global_step}",
            f"epoch={event.epoch:.4f}",
            f"session={event.session_id}",
        ]
        if event.loss is not None:
            parts.append(f"loss={event.loss:.8f}")
        if event.error:
            parts.append(f"error={event.error}")
        if event.message:
            parts.append(f"message={event.message}")
        _LOG.info(" | ".join(parts))
