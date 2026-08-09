"""Scorecard rendering helpers."""

from __future__ import annotations

from aiodoo_training.system_training_contract.quality.common import GateOutcome
from aiodoo_training.system_training_contract.quality.harness import QualityReport

# GateOutcome re-exported via common; harness imports it — fix circular import
# by keeping render here and importing QualityReport lazily in render.


def render_scorecard(report: QualityReport) -> str:
    lines = [
        f"FP2 Corpus Quality Scorecard — {report.corpus_root}",
        f"Native records: {report.total_native_records}  Projection: {report.total_projection_records}",
        f"Readiness: {report.readiness}",
        "",
        "Gates:",
    ]
    for name, outcome in sorted(report.gates.items()):
        lines.append(f"  [{outcome}] {name}")
    if report.readiness_reasons:
        lines.append("")
        lines.append("Notes:")
        for r in report.readiness_reasons:
            lines.append(f"  - {r}")
    lines.append("")
    lines.append(
        f"Coverage: {report.coverage.get('covered_count')}/"
        f"{report.coverage.get('preferred_total')} "
        f"({report.coverage.get('coverage_pct')}%)"
    )
    lines.append(
        f"Domain: odoo={report.domain.get('odoo')} generic={report.domain.get('generic')} "
        f"({report.domain.get('odoo_pct')}% odoo)"
    )
    lines.append(
        f"Duplicates: groups={report.duplicates.get('duplicate_groups')}"
    )
    neg_ok = sum(1 for n in report.negatives if n.get("matched"))
    lines.append(f"Negatives matched: {neg_ok}/{len(report.negatives)}")
    return "\n".join(lines) + "\n"
