"""TR-4 FP2 corpus quality harness.

Evaluates fixture corpora for scale readiness. Does not train adapters.
"""

from __future__ import annotations

from aiodoo_training.system_training_contract.quality.common import GateOutcome
from aiodoo_training.system_training_contract.quality.harness import (
    QualityReport,
    evaluate_fp2_corpus,
    write_report,
)
from aiodoo_training.system_training_contract.quality.scorecard import render_scorecard
from aiodoo_training.system_training_contract.quality.splits import (
    SplitAssignment,
    assign_split,
    document_split_strategy,
)

QualityGate = GateOutcome

__all__ = [
    "GateOutcome",
    "QualityGate",
    "QualityReport",
    "evaluate_fp2_corpus",
    "write_report",
    "render_scorecard",
    "SplitAssignment",
    "assign_split",
    "document_split_strategy",
]
