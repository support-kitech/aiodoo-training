#!/usr/bin/env python3
"""Validate an experiment configuration (compose, validate, resolve, fingerprint id)."""

from __future__ import annotations

import argparse
from pathlib import Path

from aiodoo_training.cli import run
from aiodoo_training.cli.commands import cmd_validate_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an aiodoo-training experiment config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML.")
    args = parser.parse_args(argv)
    return run(lambda: cmd_validate_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
