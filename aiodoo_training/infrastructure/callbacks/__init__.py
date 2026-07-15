"""Training callback plugins."""

from aiodoo_training.infrastructure.callbacks.logging import LoggingCallback
from aiodoo_training.infrastructure.callbacks.null import NullCallback, register_default_callbacks

__all__ = [
    "LoggingCallback",
    "NullCallback",
    "register_default_callbacks",
]
