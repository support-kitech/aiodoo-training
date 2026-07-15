"""CLI helpers shared by repository-root entry scripts."""

from aiodoo_training.cli.commands import (
    CommandBuilder,
    CommandContext,
    CommandRegistry,
    build_default_registry,
    cmd_doctor,
    cmd_fingerprint,
    cmd_train,
    cmd_validate_config,
)
from aiodoo_training.cli.runtime import run

__all__ = [
    "CommandBuilder",
    "CommandContext",
    "CommandRegistry",
    "build_default_registry",
    "cmd_doctor",
    "cmd_fingerprint",
    "cmd_train",
    "cmd_validate_config",
    "run",
]
