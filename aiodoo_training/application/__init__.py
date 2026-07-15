"""Application layer: public orchestration that wires configs into frozen pipeline."""

from aiodoo_training.application.train_orchestrator import (
    ExecutionResult,
    emit_execution_result,
    run_train_from_config,
    train_exit_code,
)

__all__ = [
    "ExecutionResult",
    "emit_execution_result",
    "run_train_from_config",
    "train_exit_code",
]
