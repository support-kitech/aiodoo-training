"""CLI: emit FP2-native fixture corpora (TR-3).

Default output: sibling aiodoo-datasets/datasets/fp2 when present.
Does not touch historical production JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiodoo_training.system_training_contract.generators.emit import emit_all_fixtures


def _default_output() -> Path:
    here = Path(__file__).resolve()
    # .../aiodoo-training/aiodoo_training/system_training_contract/generators/cli.py
    workspace = here.parents[4]
    candidate = workspace / "aiodoo-datasets" / "datasets" / "fp2"
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit FP2-native Training fixtures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output(),
        help="Directory for fp2 fixture JSONL (default: aiodoo-datasets/datasets/fp2)",
    )
    args = parser.parse_args(argv)
    counts = emit_all_fixtures(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
