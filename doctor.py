#!/usr/bin/env python3
"""Print environment / repository diagnostic information."""

from __future__ import annotations

from aiodoo_training.cli import run
from aiodoo_training.cli.commands import cmd_doctor


def main() -> int:
    return run(cmd_doctor)


if __name__ == "__main__":
    raise SystemExit(main())
