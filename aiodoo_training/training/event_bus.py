"""In-process training event bus — synchronous ordered dispatch."""

from __future__ import annotations

from collections.abc import Callable

from aiodoo_training.domain.training_events import TrainingEvent
from aiodoo_training.ports.callback import TrainingCallback
from aiodoo_training.training.context import CallbackContext, TrainingContext


class TrainingEventBus:
    """Publish training events to subscribers in registration order."""

    def __init__(self) -> None:
        self._subscribers: list[TrainingCallback] = []

    def subscribe(self, callback: TrainingCallback) -> None:
        """Register a callback for ordered synchronous dispatch."""
        self._subscribers.append(callback)

    def subscribe_fn(self, fn: Callable[[TrainingEvent, CallbackContext], None]) -> None:
        """Register a plain callable wrapped as a :class:`TrainingCallback`."""

        class _FnCallback(TrainingCallback):
            def on_event(self, event: TrainingEvent, context: CallbackContext) -> None:
                fn(event, context)

        self.subscribe(_FnCallback())

    @property
    def subscribers(self) -> tuple[TrainingCallback, ...]:
        return tuple(self._subscribers)

    def publish(self, event: TrainingEvent, context: TrainingContext) -> None:
        """Dispatch ``event`` to all subscribers synchronously."""
        callback_context = CallbackContext(training_context=context)
        for subscriber in self._subscribers:
            subscriber.on_event(event, callback_context)

        if context.metric_collector is not None:
            context.metric_collector.on_event(event, callback_context)

        if context.tracker is not None and event.metrics:
            context.tracker.log_metrics(event.metrics)
