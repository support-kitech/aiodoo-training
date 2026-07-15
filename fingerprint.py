#!/usr/bin/env python3
"""Print the deterministic experiment fingerprint for a config."""

from __future__ import annotations

import argparse
from pathlib import Path

from aiodoo_training.cli import run
from aiodoo_training.cli.commands import cmd_fingerprint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fingerprint an aiodoo-training experiment config.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML.")
    args = parser.parse_args(argv)
    return run(lambda: cmd_fingerprint(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
