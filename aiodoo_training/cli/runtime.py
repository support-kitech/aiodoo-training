"""Shared runtime helpers for repository-root entry scripts."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

logger = logging.getLogger("aiodoo_training.cli")


def _ensure_logging() -> None:
    """Attach a stderr handler when the process has none (CLI entry)."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def run(command: Callable[[], int]) -> int:
    """
    Execute a command callable and map domain/errors to process exit codes.

    Exit codes:
        0 — success
        1 — domain / configuration error
        2 — not implemented (deferred phase)

    Unexpected failures keep the CLI exit contract while logging full tracebacks.
    """
    from aiodoo_training.exceptions import AiodooTrainingError

    _ensure_logging()
    try:
        return command()
    except NotImplementedError as exc:
        logger.error("Not implemented: %s", exc)
        print(f"Not implemented: {exc}", file=sys.stderr)
        return 2
    except AiodooTrainingError as exc:
        logger.exception("Command failed with domain error")
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — map uncaught errors to exit 1
        logger.exception("Command failed with unexpected error")
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
