"""CLI: run TR-4 FP2 corpus quality evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiodoo_training.system_training_contract.quality.harness import (
    evaluate_fp2_corpus,
    write_report,
)
from aiodoo_training.system_training_contract.quality.scorecard import render_scorecard


def _default_corpus() -> Path:
    here = Path(__file__).resolve()
    # .../aiodoo_training/system_training_contract/quality/cli.py
    training_root = here.parents[3]
    return training_root / "fixtures" / "fp2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate FP2-native corpus quality (TR-4)")
    parser.add_argument("--corpus", type=Path, default=_default_corpus())
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write JSON report path (default: corpus/quality_report_tr4.json)",
    )
    args = parser.parse_args(argv)
    report_path = args.report or (args.corpus / "quality_report_tr4.json")
    report = evaluate_fp2_corpus(args.corpus)
    write_report(report, report_path)
    print(render_scorecard(report))
    print(json.dumps({"readiness": report.readiness, "report": str(report_path)}, indent=2))
    return 0 if report.readiness == "READY_FOR_TR5" else 2


if __name__ == "__main__":
    raise SystemExit(main())
