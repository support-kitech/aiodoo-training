"""Null training callback — no-op event sink."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiodoo_training.ports.callback import TrainingCallback

if TYPE_CHECKING:
    from aiodoo_training.domain.training_events import TrainingEvent
    from aiodoo_training.training.context import CallbackContext


class NullCallback(TrainingCallback):
    """Discard all training events."""

    def on_event(self, event: TrainingEvent, context: CallbackContext) -> None:
        _ = (event, context)


def register_default_callbacks(*, overwrite: bool = False) -> None:
    """Register null and logging callbacks."""
    from aiodoo_training.infrastructure.callbacks.logging import LoggingCallback
    from aiodoo_training.registries import callback_registry

    mappings: dict[str, type[TrainingCallback]] = {
        "null": NullCallback,
        "logging": LoggingCallback,
    }
    for key, cls in mappings.items():
        if not callback_registry.exists(key) or overwrite:
            callback_registry.register(key, cls, overwrite=overwrite)
