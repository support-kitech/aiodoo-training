"""TR-6 FP2 corpus evaluation and training-pack readiness.

Evaluates controlled_batch_1 for adapter-training suitability.
Does not train adapters. Does not generate new corpora.
"""

from __future__ import annotations

from aiodoo_training.system_training_contract.evaluation.harness import (
    TrainingReadiness,
    Tr6Report,
    evaluate_controlled_batch,
    write_tr6_report,
)
from aiodoo_training.system_training_contract.evaluation.scorecard import render_tr6_scorecard

__all__ = [
    "TrainingReadiness",
    "Tr6Report",
    "evaluate_controlled_batch",
    "write_tr6_report",
    "render_tr6_scorecard",
]
