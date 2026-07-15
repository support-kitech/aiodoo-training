#!/usr/bin/env python3
"""Resume training from a checkpoint (Phase 0: not implemented)."""

from __future__ import annotations

import argparse
from pathlib import Path

from aiodoo_training.cli import run
from aiodoo_training.cli.commands import cmd_resume


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resume an aiodoo-training run.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint directory.",
    )
    args = parser.parse_args(argv)
    return run(lambda: cmd_resume(args.checkpoint))


if __name__ == "__main__":
    raise SystemExit(main())
