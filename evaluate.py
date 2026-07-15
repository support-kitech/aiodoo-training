#!/usr/bin/env python3
"""Evaluate a trained artifact (Phase 0: not implemented)."""

from __future__ import annotations

import argparse
from pathlib import Path

from aiodoo_training.cli import run
from aiodoo_training.cli.commands import cmd_evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate an aiodoo-training artifact.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML.")
    args = parser.parse_args(argv)
    return run(lambda: cmd_evaluate(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
