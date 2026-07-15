#!/usr/bin/env python3
"""
AIODOO Training orchestrator entrypoint.

Validates configuration and, unless ``--validate-only`` is set, runs the
public training pipeline (same path as ``train.py``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aiodoo_training.cli import run
from aiodoo_training.cli.commands import cmd_train, cmd_validate_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and run an aiodoo-training experiment pipeline.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration; do not train.",
    )
    args = parser.parse_args(argv)

    def _command() -> int:
        code = cmd_validate_config(args.config)
        if code != 0:
            return code
        if args.validate_only:
            return 0
        return cmd_train(args.config)

    return run(_command)


if __name__ == "__main__":
    raise SystemExit(main())
