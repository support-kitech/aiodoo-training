"""FP2 corpus quality harness — evaluate fixtures for TR-5 readiness."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiodoo_training.system_training_contract.quality.analysis import (
    analyze_feedback_operation_vs_objective,
    analyze_loop_decisions,
    analyze_planning,
    analyze_state_isolation,
    analyze_work_units,
    capability_coverage,
    domain_distribution,
    find_duplicates,
)
from aiodoo_training.system_training_contract.quality.common import (
    NATIVE_FAMILIES,
    GateOutcome,
    load_jsonl,
    stable_dumps,
)
from aiodoo_training.system_training_contract.quality.gates import (
    provenance_ok_for_projected,
    scan_forbidden_how,
    scan_taxonomy,
)
from aiodoo_training.system_training_contract.quality.negatives import (
    NEGATIVE_CASES,
    evaluate_negative_case,
)
from aiodoo_training.system_training_contract.quality.splits import (
    assign_split,
    document_split_strategy,
)
from aiodoo_training.system_training_contract.records import (
    TrainingRecordError,
    validate_record_mapping,
)
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

# Re-export for callers expecting QualityGate name
QualityGate = GateOutcome


@dataclass
class QualityReport:
    corpus_root: str
    total_native_records: int = 0
    total_projection_records: int = 0
    per_family: dict[str, int] = field(default_factory=dict)
    schema_failures: list[str] = field(default_factory=list)
    how_violations: list[str] = field(default_factory=list)
    taxonomy_violations: list[str] = field(default_factory=list)
    serialization_failures: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    domain: dict[str, Any] = field(default_factory=dict)
    duplicates: dict[str, Any] = field(default_factory=dict)
    work_units: dict[str, Any] = field(default_factory=dict)
    planning: dict[str, Any] = field(default_factory=dict)
    feedback: dict[str, Any] = field(default_factory=dict)
    continuity: dict[str, Any] = field(default_factory=dict)
    loops: dict[str, Any] = field(default_factory=dict)
    negatives: list[dict[str, Any]] = field(default_factory=list)
    split_strategy: dict[str, Any] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    gates: dict[str, str] = field(default_factory=dict)
    readiness: str = "NOT_READY"
    readiness_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_root": self.corpus_root,
            "total_native_records": self.total_native_records,
            "total_projection_records": self.total_projection_records,
            "per_family": self.per_family,
            "schema_failures": self.schema_failures,
            "how_violations": self.how_violations,
            "taxonomy_violations": self.taxonomy_violations,
            "serialization_failures": self.serialization_failures,
            "coverage": self.coverage,
            "domain": self.domain,
            "duplicates": {
                "unique_fingerprints": self.duplicates.get("unique_fingerprints"),
                "duplicate_groups": self.duplicates.get("duplicate_groups"),
                # omit huge id maps in default dict; keep counts
            },
            "duplicate_record_ids": self.duplicates.get("duplicate_record_ids", {}),
            "work_units": self.work_units,
            "planning": self.planning,
            "feedback": self.feedback,
            "continuity": self.continuity,
            "loops": self.loops,
            "negatives": self.negatives,
            "split_strategy": self.split_strategy,
            "split_counts": self.split_counts,
            "gates": self.gates,
            "readiness": self.readiness,
            "readiness_reasons": self.readiness_reasons,
            "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
        }


def evaluate_fp2_corpus(corpus_root: str | Path) -> QualityReport:
    root = Path(corpus_root)
    report = QualityReport(corpus_root=str(root))
    native: list[dict[str, Any]] = []
    for fam in NATIVE_FAMILIES:
        rows = load_jsonl(root / f"{fam}.jsonl")
        report.per_family[fam] = len(rows)
        native.extend(rows)
    report.total_native_records = len(native)

    proj = load_jsonl(root / "projection_fixtures.jsonl")
    report.total_projection_records = len(proj)

    # Schema + HOW + taxonomy + determinism
    for rec in native:
        rid = str(rec.get("record_id") or "?")
        try:
            validated = validate_record_mapping(rec)
            if stable_dumps(validated) != stable_dumps(validate_record_mapping(validated)):
                report.serialization_failures.append(rid)
        except TrainingRecordError as exc:
            report.schema_failures.append(f"{rid}:{exc}")
            continue
        for issue in scan_forbidden_how(rec):
            report.how_violations.append(f"{rid}:{issue}")
        for issue in scan_taxonomy(rec):
            report.taxonomy_violations.append(f"{rid}:{issue}")
        # Projected native records (if any) need provenance
        meta = rec.get("metadata") if isinstance(rec.get("metadata"), Mapping) else {}
        if meta.get("projected"):
            for issue in provenance_ok_for_projected(rec):
                report.schema_failures.append(f"{rid}:{issue}")

    # Projection fixtures: provenance required on result envelope
    for row in proj:
        prov = row.get("provenance") if isinstance(row.get("provenance"), Mapping) else {}
        if not prov.get("source_record_id") or not prov.get("projection_status"):
            report.schema_failures.append(
                f"projection:{row.get('fixture_id')}:incomplete_provenance"
            )

    report.coverage = capability_coverage(native)
    report.domain = domain_distribution(native)
    report.duplicates = find_duplicates(native)
    report.work_units = analyze_work_units(native)
    report.planning = analyze_planning(native)
    report.feedback = analyze_feedback_operation_vs_objective(native)
    report.continuity = analyze_state_isolation(native)
    report.loops = analyze_loop_decisions(native)
    report.negatives = [evaluate_negative_case(c) for c in NEGATIVE_CASES]
    report.split_strategy = document_split_strategy()
    split_counter: Counter[str] = Counter()
    for rec in native:
        split_counter[assign_split(rec).value] += 1
    report.split_counts = dict(split_counter)

    report.gates = _compute_gates(report)
    report.readiness, report.readiness_reasons = _readiness(report)
    return report


def _compute_gates(report: QualityReport) -> dict[str, str]:
    gates: dict[str, str] = {}
    expected_native = 104  # classic TR-3 inventory; TR-5.2 may expand fixtures
    if report.total_native_records == expected_native and report.total_projection_records == 7:
        gates["fixture_inventory"] = GateOutcome.PASS.value
    elif 100 <= report.total_native_records <= 200 and report.total_projection_records == 7:
        # Expanded golden fixtures (coverage gaps) remain fixture-scale.
        gates["fixture_inventory"] = GateOutcome.PASS.value
    elif report.total_native_records >= 80:
        gates["fixture_inventory"] = GateOutcome.WARN.value
    else:
        gates["fixture_inventory"] = GateOutcome.FAIL.value

    gates["schema"] = (
        GateOutcome.PASS.value if not report.schema_failures else GateOutcome.FAIL.value
    )
    gates["forbidden_how"] = (
        GateOutcome.PASS.value if not report.how_violations else GateOutcome.FAIL.value
    )
    gates["taxonomy"] = (
        GateOutcome.PASS.value if not report.taxonomy_violations else GateOutcome.FAIL.value
    )
    gates["serialization"] = (
        GateOutcome.PASS.value
        if not report.serialization_failures
        else GateOutcome.FAIL.value
    )
    cov = report.coverage.get("coverage_pct", 0)
    if cov >= 70:
        gates["capability_coverage"] = GateOutcome.PASS.value
    elif cov >= 40:
        gates["capability_coverage"] = GateOutcome.WARN.value
    else:
        gates["capability_coverage"] = GateOutcome.FAIL.value

    gates["work_unit"] = GateOutcome.PASS.value if report.work_units.get("ok") else GateOutcome.FAIL.value
    gates["planning"] = GateOutcome.PASS.value if report.planning.get("ok") else GateOutcome.FAIL.value
    gates["feedback"] = GateOutcome.PASS.value if report.feedback.get("ok") else GateOutcome.FAIL.value
    gates["continuity"] = GateOutcome.PASS.value if report.continuity.get("ok") else GateOutcome.FAIL.value
    gates["loop_decisions"] = GateOutcome.PASS.value if report.loops.get("ok") else GateOutcome.FAIL.value
    gates["duplicates"] = (
        GateOutcome.PASS.value
        if report.duplicates.get("duplicate_groups", 0) == 0
        else GateOutcome.WARN.value
    )
    neg_ok = all(n.get("matched") for n in report.negatives)
    gates["negative_corpus"] = GateOutcome.PASS.value if neg_ok else GateOutcome.FAIL.value

    odoo_pct = report.domain.get("odoo_pct", 0)
    if 15 <= odoo_pct <= 85:
        gates["odoo_generic_balance"] = GateOutcome.PASS.value
    elif 5 <= odoo_pct <= 95:
        gates["odoo_generic_balance"] = GateOutcome.WARN.value
    else:
        gates["odoo_generic_balance"] = GateOutcome.FAIL.value

    gates["split_strategy"] = GateOutcome.PASS.value if report.split_strategy else GateOutcome.FAIL.value
    return gates


def _readiness(report: QualityReport) -> tuple[str, list[str]]:
    """
    READY_FOR_TR5 requires all hard gates PASS and coverage WARN+.

    Hard gates: schema, forbidden_how, taxonomy, serialization, work_unit,
    planning, feedback, continuity, loop_decisions, negative_corpus,
    fixture_inventory.
    Soft: capability_coverage may be WARN; duplicates WARN ok.
    """
    hard = (
        "fixture_inventory",
        "schema",
        "forbidden_how",
        "taxonomy",
        "serialization",
        "work_unit",
        "planning",
        "feedback",
        "continuity",
        "loop_decisions",
        "negative_corpus",
    )
    reasons: list[str] = []
    for g in hard:
        if report.gates.get(g) != GateOutcome.PASS.value:
            reasons.append(f"hard_gate_failed:{g}={report.gates.get(g)}")
    if report.gates.get("capability_coverage") == GateOutcome.FAIL.value:
        reasons.append("capability_coverage_fail")
    if report.gates.get("odoo_generic_balance") == GateOutcome.FAIL.value:
        reasons.append("odoo_generic_balance_fail")

    # Explicit TR-4 policy: WARN coverage is acceptable for fixture-scale TR-5
    # only if uncovered capabilities are documented — still READY with WARN note.
    if report.gates.get("capability_coverage") == GateOutcome.WARN.value:
        reasons.append(
            "warn:capability_coverage_incomplete:"
            f"{report.coverage.get('uncovered_count')} uncovered preferred IDs"
        )

    if any(r.startswith("hard_gate") or r.endswith("_fail") for r in reasons):
        return "NOT_READY", reasons
    uncovered = int(report.coverage.get("uncovered_count") or 0)
    if uncovered:
        reasons.append(
            f"tr5_prerequisite:expand_coverage_for_{uncovered}_uncovered_preferred_ids:"
            + ",".join(report.coverage.get("uncovered") or [])
        )
    return "READY_FOR_TR5", reasons or ["all_hard_gates_pass"]


def write_report(report: QualityReport, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
