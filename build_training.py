#!/usr/bin/env python3
"""
AIODOO Training orchestrator entrypoint.

Phase 0 validates the experiment config and stops before any training logic.
Later phases will compose the full TrainingPipeline here (similar to
aiodoo-datasets/build_dataset.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aiodoo_training.cli import run
from aiodoo_training.cli.commands import cmd_validate_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and run an aiodoo-training experiment pipeline.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration (Phase 0 default behavior).",
    )
    args = parser.parse_args(argv)

    def _command() -> int:
        code = cmd_validate_config(args.config)
        if code != 0:
            return code
        if args.validate_only:
            return 0
        print(
            "Training pipeline is not implemented in Phase 0. "
            "Re-run with --validate-only or wait for later phases.",
            file=sys.stderr,
        )
        return 2

    return run(_command)


if __name__ == "__main__":
    raise SystemExit(main())
