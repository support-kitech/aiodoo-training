#!/usr/bin/env python3
"""Merge a PEFT adapter into base weights (Phase 0: not implemented)."""

from __future__ import annotations

import argparse
from pathlib import Path

from aiodoo_training.cli import run
from aiodoo_training.cli.commands import cmd_merge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge an aiodoo-training adapter.")
    parser.add_argument("--adapter", type=Path, required=True, help="Path to adapter directory.")
    args = parser.parse_args(argv)
    return run(lambda: cmd_merge(args.adapter))


if __name__ == "__main__":
    raise SystemExit(main())
