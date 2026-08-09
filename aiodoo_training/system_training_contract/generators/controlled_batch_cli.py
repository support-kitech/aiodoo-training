"""CLI: emit TR-5 controlled FP2 batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiodoo_training.system_training_contract.generators.controlled_batch import (
    emit_controlled_batch,
)


def _default_output() -> Path:
    here = Path(__file__).resolve()
    workspace = here.parents[4]
    return workspace / "aiodoo-datasets" / "datasets" / "fp2" / "controlled_batch_1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit TR-5 controlled FP2 batch")
    parser.add_argument("--output-dir", type=Path, default=_default_output())
    parser.add_argument("--target", type=int, default=1200)
    args = parser.parse_args(argv)
    result = emit_controlled_batch(args.output_dir, target=args.target)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.decision in {"PASS", "PASS_WITH_WARNINGS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
