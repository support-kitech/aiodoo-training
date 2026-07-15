#!/usr/bin/env python3
"""
Compose, validate, resolve, and print an experiment configuration.

Useful for inspecting include-merged YAML without training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiodoo_training.cli import run
from aiodoo_training.config import ConfigSystem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compose and dump a resolved aiodoo-training config.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML.")
    args = parser.parse_args(argv)

    def _command() -> int:
        system = ConfigSystem()
        model, experiment_id, resolved = system.load_experiment(args.config)
        print(f"name: {model.name}")
        print(f"schema_version: {model.schema_version}")
        print(f"config_experiment_id: {experiment_id.value}")
        print("--- resolved ---")
        print(json.dumps(resolved, indent=2, sort_keys=True, default=str))
        return 0

    return run(_command)


if __name__ == "__main__":
    raise SystemExit(main())
