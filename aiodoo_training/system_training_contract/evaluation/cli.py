"""CLI: TR-6 evaluate controlled FP2 batch for training readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiodoo_training.system_training_contract.evaluation.harness import (
    TrainingReadiness,
    evaluate_controlled_batch,
    write_tr6_report,
)
from aiodoo_training.system_training_contract.evaluation.scorecard import render_tr6_scorecard


def _default_corpus() -> Path:
    here = Path(__file__).resolve()
    workspace = here.parents[4]
    return workspace / "aiodoo-datasets" / "datasets" / "fp2" / "controlled_batch_1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TR-6 evaluate FP2 controlled batch")
    parser.add_argument("--corpus", type=Path, default=_default_corpus())
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    report_path = args.report or (args.corpus / "quality_report_tr6.json")
    report = evaluate_controlled_batch(args.corpus)
    write_tr6_report(report, report_path)
    print(render_tr6_scorecard(report))
    print(json.dumps({"readiness": report.readiness, "report": str(report_path)}, indent=2))
    if report.readiness == TrainingReadiness.READY_FOR_TRAINING.value:
        return 0
    if report.readiness == TrainingReadiness.READY_WITH_REQUIRED_DATA_FIXES.value:
        return 0  # evaluation succeeded; readiness is conditional
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
