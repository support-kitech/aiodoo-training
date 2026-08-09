"""Forbidden HOW vocabulary for model-facing Training records (TR-2).

Synchronized from ``aiodoo.intelligence.model_contract`` — Training must not
teach models to emit these as Engineering WHAT.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "FORBIDDEN_IMPL_IDS",
    "FORBIDDEN_BACKEND_ACTIONS",
    "FORBIDDEN_ARG_KEYS",
    "assert_no_forbidden_how",
]

FORBIDDEN_IMPL_IDS: frozenset[str] = frozenset(
    {
        "local_workspace",
        "local_git",
        "local_program",
        "local_http",
        "local_diagnostics",
        "local_artifact",
        "local_validation",
        "local_snapshot",
        "local_repair",
    }
)

FORBIDDEN_BACKEND_ACTIONS: frozenset[str] = frozenset(
    {
        "shell",
        "bash",
        "sh",
        "git",
        "python",
        "subprocess",
        "docker",
        "kubectl",
        "mcp",
        "ssh",
        "curl",
        "wget",
        "pytest",
        "ruff",
        "mypy",
        "pyright",
        "unittest",
        "zip",
        "cp",
        "filesystem",
        "cmd",
        "powershell",
        "sed",
        "python_fix",
        "pytest_fix",
        "ruff_fix",
        "shell_fix",
        "git_patch",
        "subprocess_fix",
        "local_repair",
        "create_file",
        "update_file",
        "delete_file",
        "apply_artifact",
        "mkdir",
        "read_file",
    }
)

FORBIDDEN_ARG_KEYS: frozenset[str] = frozenset(
    {
        "command",
        "shell",
        "bash",
        "subprocess",
        "argv",
        "git_command",
        "implementation_id",
        "backend",
    }
)


class ForbiddenHowError(ValueError):
    """Raised when a Training record contains System HOW leakage."""


def assert_no_forbidden_how(
    *,
    capability_id: str | None = None,
    args: Mapping[str, Any] | None = None,
    text_fields: Mapping[str, str] | None = None,
) -> None:
    """Fail closed if HOW vocabulary appears in model-facing fields."""
    issues: list[str] = []
    if capability_id:
        key = capability_id.strip().lower()
        if key in FORBIDDEN_IMPL_IDS or key.startswith("local_"):
            issues.append(f"forbidden implementation id as capability: {capability_id!r}")
        if key in FORBIDDEN_BACKEND_ACTIONS:
            issues.append(f"forbidden backend/action as capability: {capability_id!r}")
    if args:
        for k in args:
            if str(k).strip().lower() in FORBIDDEN_ARG_KEYS:
                issues.append(f"forbidden arg key: {k!r}")
    if text_fields:
        for name, value in text_fields.items():
            blob = (value or "").strip().lower()
            for token in FORBIDDEN_IMPL_IDS:
                if token in blob:
                    issues.append(f"{name} contains forbidden impl token {token!r}")
    if issues:
        raise ForbiddenHowError("; ".join(issues))
