#!/usr/bin/env python3
"""Run the aiodoo-training test suite from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    pytest_args = argv if argv is not None else sys.argv[1:]
    if not pytest_args:
        pytest_args = ["-q"]

    return int(pytest.main(pytest_args))


if __name__ == "__main__":
    raise SystemExit(main())
