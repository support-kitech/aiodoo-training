"""Register default formatters into the formatter registry."""

from __future__ import annotations

from aiodoo_training.datasets.formatters.formatters import (
    ApprovalFormatter,
    CodingFormatter,
    ContextFormatter,
    ConversationFormatter,
    EvaluationFormatter,
    ExecutionFormatter,
    PlannerFormatter,
    RepairFormatter,
)
from aiodoo_training.registries import formatter_registry

_DEFAULTS = {
    "planner": PlannerFormatter,
    "coding": CodingFormatter,
    "repair": RepairFormatter,
    "context": ContextFormatter,
    "execution": ExecutionFormatter,
    "approval": ApprovalFormatter,
    "conversation": ConversationFormatter,
    "evaluation": EvaluationFormatter,
}


def register_default_formatters(*, overwrite: bool = False) -> None:
    """Register built-in protocol formatters (idempotent with overwrite)."""
    for key, cls in _DEFAULTS.items():
        if formatter_registry.exists(key) and not overwrite:
            continue
        formatter_registry.register(key, cls, overwrite=overwrite)
