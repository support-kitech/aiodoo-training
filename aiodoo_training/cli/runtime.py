"""Shared runtime helpers for repository-root entry scripts."""

from __future__ import annotations

import sys
from collections.abc import Callable


def run(command: Callable[[], int]) -> int:
    """
    Execute a command callable and map domain/errors to process exit codes.

    Exit codes:
        0 — success
        1 — domain / configuration error
        2 — not implemented (deferred phase)
    """
    from aiodoo_training.exceptions import AiodooTrainingError

    try:
        return command()
    except NotImplementedError as exc:
        print(f"Not implemented: {exc}", file=sys.stderr)
        return 2
    except AiodooTrainingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
