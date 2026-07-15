#!/usr/bin/env python3
"""Run a training experiment (Phase 0: not implemented)."""

from __future__ import annotations

import argparse
from pathlib import Path

from aiodoo_training.cli import run
from aiodoo_training.cli.commands import cmd_train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an aiodoo-training experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML.")
    args = parser.parse_args(argv)
    return run(lambda: cmd_train(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
