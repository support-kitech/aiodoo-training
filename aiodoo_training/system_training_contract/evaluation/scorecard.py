"""Render TR-6 scorecard text."""

from __future__ import annotations

from aiodoo_training.system_training_contract.evaluation.harness import Tr6Report


def render_tr6_scorecard(report: Tr6Report) -> str:
    lines = [
        f"TR-6 Training-Pack Readiness — {report.corpus_root}",
        f"Native records: {report.inventory.get('total_native')}",
        f"Dev pack: {report.inventory.get('development_pack')}  "
        f"Reasoning pack: {report.inventory.get('reasoning_pack')}",
        f"Checksum OK: {report.checksum_ok}",
        f"Readiness: {report.readiness}",
        "",
        "Hard gates:",
    ]
    for k, v in sorted(report.hard_gates.items()):
        lines.append(f"  [{v}] {k}")
    lines.append("")
    lines.append("Soft metrics:")
    for k, v in sorted(report.soft_metrics.items()):
        lines.append(f"  [{v}] {k}")
    lines.append("")
    lines.append(
        f"Diversity: unique_families={report.diversity.get('unique_families')} "
        f"concentration={report.diversity.get('concentration_pct')}% "
        f"largest={report.diversity.get('largest_family')}"
    )
    lines.append(
        f"Odoo/generic: {report.odoo.get('odoo')}/{report.odoo.get('generic')} "
        f"({report.odoo.get('odoo_pct')}%) ambiguous={report.odoo.get('ambiguous')}"
    )
    if report.readiness_rationale:
        lines.append("")
        lines.append("Rationale:")
        for r in report.readiness_rationale:
            lines.append(f"  - {r}")
    if report.issues.get("P0") or report.issues.get("P1"):
        lines.append("")
        lines.append("Blocking / required fixes:")
        for sev in ("P0", "P1"):
            for item in report.issues.get(sev) or []:
                lines.append(f"  [{sev}] {item}")
    return "\n".join(lines) + "\n"
