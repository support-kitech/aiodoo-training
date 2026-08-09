"""TR-6/TR-7 evaluation harness — training-pack readiness for controlled batches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from aiodoo_training.system_training_contract.evaluation.metrics import (
    analyze_capability_distribution,
    analyze_continuity,
    analyze_negative_contamination,
    analyze_objective_completion,
    analyze_odoo_quality,
    analyze_packs,
    analyze_provider_separation,
    analyze_scenario_diversity,
    analyze_splits,
    load_native_records,
    load_pack,
)
from aiodoo_training.system_training_contract.quality.common import NATIVE_FAMILIES, GateOutcome
from aiodoo_training.system_training_contract.quality.gates import (
    scan_forbidden_how,
    scan_taxonomy,
)
from aiodoo_training.system_training_contract.records import (
    TrainingRecordError,
    validate_record_mapping,
)
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION


class TrainingReadiness(StrEnum):
    READY_FOR_TRAINING = "READY_FOR_TRAINING"
    READY_WITH_REQUIRED_DATA_FIXES = "READY_WITH_REQUIRED_DATA_FIXES"
    NOT_READY = "NOT_READY"


@dataclass
class Tr6Report:
    corpus_root: str
    inventory: dict[str, Any] = field(default_factory=dict)
    checksum_ok: bool = False
    hard_gates: dict[str, str] = field(default_factory=dict)
    soft_metrics: dict[str, str] = field(default_factory=dict)
    packs: dict[str, Any] = field(default_factory=dict)
    diversity: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    objective_completion: dict[str, Any] = field(default_factory=dict)
    continuity: dict[str, Any] = field(default_factory=dict)
    odoo: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)
    negatives: dict[str, Any] = field(default_factory=dict)
    provider_separation: dict[str, Any] = field(default_factory=dict)
    issues: dict[str, list[str]] = field(default_factory=dict)
    readiness: str = TrainingReadiness.NOT_READY.value
    readiness_rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
            "evaluation": "tr6",
            "corpus_root": self.corpus_root,
            "inventory": self.inventory,
            "checksum_ok": self.checksum_ok,
            "hard_gates": self.hard_gates,
            "soft_metrics": self.soft_metrics,
            "packs": {
                "development": {
                    k: v
                    for k, v in (self.packs.get("development") or {}).items()
                    if k != "issues"
                },
                "reasoning": {
                    k: v
                    for k, v in (self.packs.get("reasoning") or {}).items()
                    if k != "issues"
                },
            },
            "pack_issue_samples": {
                "development": (self.packs.get("development") or {}).get("issues", [])[:20],
                "reasoning": (self.packs.get("reasoning") or {}).get("issues", [])[:20],
            },
            "diversity": {
                k: v
                for k, v in self.diversity.items()
                if k != "family_counts"
            },
            "capabilities": {
                "underrepresented": self.capabilities.get("underrepresented"),
                "overrepresented": self.capabilities.get("overrepresented"),
                "superficial": self.capabilities.get("superficial"),
                "matrix_summary": {
                    c: {
                        "count": v["count"],
                        "families": v["families"],
                        "successish": v["successish"],
                        "failureish": v["failureish"],
                    }
                    for c, v in (self.capabilities.get("matrix") or {}).items()
                },
            },
            "objective_completion": self.objective_completion,
            "continuity": self.continuity,
            "odoo": self.odoo,
            "splits": self.splits,
            "negatives": self.negatives,
            "provider_separation": {
                "ok": self.provider_separation.get("ok"),
                "development_providers": self.provider_separation.get("development_providers"),
                "reasoning_providers": self.provider_separation.get("reasoning_providers"),
                "issue_count": len(self.provider_separation.get("issues") or []),
            },
            "issues": self.issues,
            "readiness": self.readiness,
            "readiness_rationale": self.readiness_rationale,
        }


def evaluate_controlled_batch(corpus_root: str | Path) -> Tr6Report:
    root = Path(corpus_root)
    report = Tr6Report(corpus_root=str(root))

    records = load_native_records(root)
    pack_dev = load_pack(root / "pack_development.jsonl")
    pack_rea = load_pack(root / "pack_reasoning.jsonl")
    splits_rows = load_pack(root / "splits.jsonl")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8")) if (root / "manifest.json").is_file() else {}

    # Inventory + checksum
    per_family = {fam: 0 for fam in NATIVE_FAMILIES}
    for rec in records:
        per_family[str(rec.get("record_type") or "unknown")] = per_family.get(
            str(rec.get("record_type") or "unknown"), 0
        ) + 1
    h = hashlib.sha256()
    for fam in NATIVE_FAMILIES:
        h.update((root / f"{fam}.jsonl").read_bytes())
    checksum = h.hexdigest()
    report.checksum_ok = checksum == str(manifest.get("checksum") or "")
    report.inventory = {
        "total_native": len(records),
        "per_family": per_family,
        "development_pack": len(pack_dev),
        "reasoning_pack": len(pack_rea),
        "splits": len(splits_rows),
        "manifest_total": manifest.get("total_native"),
        "manifest_decision": manifest.get("decision"),
        "checksum": checksum,
        "manifest_checksum": manifest.get("checksum"),
    }

    # Hard: schema + HOW + taxonomy on all native records
    schema_fail = how_fail = tax_fail = 0
    for rec in records:
        try:
            validate_record_mapping(rec)
        except TrainingRecordError:
            schema_fail += 1
            continue
        if scan_forbidden_how(rec):
            how_fail += 1
        if scan_taxonomy(rec):
            tax_fail += 1

    report.packs = analyze_packs(pack_dev, pack_rea)
    report.diversity = analyze_scenario_diversity(records)
    report.capabilities = analyze_capability_distribution(
        records, pack_dev=pack_dev, pack_rea=pack_rea
    )
    report.objective_completion = analyze_objective_completion(records)
    report.continuity = analyze_continuity(records)
    report.odoo = analyze_odoo_quality(records)
    report.splits = analyze_splits(records, splits_rows)
    report.negatives = analyze_negative_contamination(root, pack_dev, pack_rea)
    report.provider_separation = analyze_provider_separation(records, report.packs)

    # Hard gates
    # TR-5 batch is exactly 1200; TR-7 derivative expands continuity within band.
    batch_ver = str(manifest.get("controlled_batch_version") or manifest.get("version") or "")
    n_records = len(records)
    if n_records == 1200:
        inv_ok = True
    elif "tr7" in batch_ver.lower() or "2.0.0" in batch_ver:
        inv_ok = 1200 <= n_records <= 1800
    else:
        inv_ok = n_records == 1200
    report.hard_gates = {
        "inventory_1200": GateOutcome.PASS.value if inv_ok else GateOutcome.FAIL.value,
        "checksum": GateOutcome.PASS.value if report.checksum_ok else GateOutcome.FAIL.value,
        "schema": GateOutcome.PASS.value if schema_fail == 0 else GateOutcome.FAIL.value,
        "forbidden_how": GateOutcome.PASS.value if how_fail == 0 else GateOutcome.FAIL.value,
        "taxonomy": GateOutcome.PASS.value if tax_fail == 0 else GateOutcome.FAIL.value,
        "pack_validity": (
            GateOutcome.PASS.value
            if report.packs["development"]["malformed"] == 0
            and report.packs["reasoning"]["malformed"] == 0
            and report.packs["development"]["how_hits"] == 0
            and report.packs["reasoning"]["how_hits"] == 0
            else GateOutcome.FAIL.value
        ),
        "split_integrity": GateOutcome.PASS.value if report.splits.get("ok") else GateOutcome.FAIL.value,
        "negative_contamination": GateOutcome.PASS.value if report.negatives.get("ok") else GateOutcome.FAIL.value,
        "continuity": GateOutcome.PASS.value if report.continuity.get("ok") else GateOutcome.FAIL.value,
        "decision_context_integrity": (
            GateOutcome.PASS.value
            if not any(i.startswith("dc_") for i in report.continuity.get("issues") or [])
            else GateOutcome.FAIL.value
        ),
        "provider_separation": GateOutcome.PASS.value if report.provider_separation.get("ok") else GateOutcome.FAIL.value,
        "objective_completion_semantics": (
            GateOutcome.PASS.value if report.objective_completion.get("ok") else GateOutcome.FAIL.value
        ),
    }

    # Soft metrics
    unique = int(report.diversity.get("unique_families") or 0)
    conc = float(report.diversity.get("concentration_pct") or 100)
    # Diversity thresholds (documented):
    # PASS: >= 40 unique normalized families AND concentration < 15%
    # WARN: >= 25 families OR concentration < 25%
    # FAIL: otherwise
    if unique >= 40 and conc < 15:
        div_gate = GateOutcome.PASS.value
    elif unique >= 25 and conc < 25:
        div_gate = GateOutcome.WARN.value
    else:
        div_gate = GateOutcome.FAIL.value

    state_n = int(report.continuity.get("state_count") or 0)
    dc_n = int(report.continuity.get("dc_count") or 0)
    loop_n = int(per_family.get("loop_decision") or 0)
    # TR-7 continuity volume target: ~60–100 meaningful records per family.
    if state_n >= 60 and dc_n >= 60 and loop_n >= 60:
        cont_bal = GateOutcome.PASS.value
    elif state_n >= 40 and dc_n >= 40:
        cont_bal = GateOutcome.WARN.value
    else:
        cont_bal = GateOutcome.FAIL.value

    odoo_pct = float(report.odoo.get("odoo_pct") or 0)
    amb = int(report.odoo.get("ambiguous") or 0)
    # Ambiguous labels must be resolved/quarantined; residual ambiguous is a soft fail.
    if amb == 0 and 15 <= odoo_pct <= 50:
        domain_gate = GateOutcome.PASS.value
    elif amb < 20 and 10 <= odoo_pct <= 60:
        domain_gate = GateOutcome.WARN.value
    else:
        domain_gate = GateOutcome.FAIL.value

    under = report.capabilities.get("underrepresented") or []
    over = report.capabilities.get("overrepresented") or []
    if len(under) == 0 and len(over) <= 6:
        bal_gate = GateOutcome.PASS.value
    elif len(under) <= 4:
        bal_gate = GateOutcome.WARN.value
    else:
        bal_gate = GateOutcome.FAIL.value

    # Pack balance: both packs non-empty and not wildly skewed vs native
    dev_n = len(pack_dev)
    rea_n = len(pack_rea)
    if 700 <= dev_n <= 1400 and 700 <= rea_n <= 1400:
        pack_bal = GateOutcome.PASS.value
    elif dev_n >= 400 and rea_n >= 400:
        pack_bal = GateOutcome.WARN.value
    else:
        pack_bal = GateOutcome.FAIL.value

    # Edge-case: failureish coverage across caps
    matrix = report.capabilities.get("matrix") or {}
    caps_with_fail = sum(1 for v in matrix.values() if int(v.get("failureish") or 0) > 0)
    if caps_with_fail >= 10:
        edge_gate = GateOutcome.PASS.value
    elif caps_with_fail >= 5:
        edge_gate = GateOutcome.WARN.value
    else:
        edge_gate = GateOutcome.FAIL.value

    # Train-only critical caps
    train_only = report.splits.get("capabilities_train_only") or []
    if len(train_only) == 0:
        split_bal = GateOutcome.PASS.value
    elif len(train_only) <= 5:
        split_bal = GateOutcome.WARN.value
    else:
        split_bal = GateOutcome.FAIL.value

    report.soft_metrics = {
        "scenario_diversity": div_gate,
        "capability_balance": bal_gate,
        "domain_balance": domain_gate,
        "pack_balance": pack_bal,
        "continuity_volume": cont_bal,
        "edge_case_coverage": edge_gate,
        "split_capability_balance": split_bal,
        "repetition": (
            GateOutcome.WARN.value
            if report.diversity.get("repetitive_families")
            else GateOutcome.PASS.value
        ),
    }

    # Issues classified
    p0: list[str] = []
    p1: list[str] = []
    p2: list[str] = []
    p3: list[str] = []
    for g, v in report.hard_gates.items():
        if v == GateOutcome.FAIL.value:
            p0.append(f"hard_gate_failed:{g}")

    if div_gate == GateOutcome.FAIL.value:
        p1.append(
            "scenario_diversity_insufficient:"
            f"unique={unique},concentration={conc}%"
        )
    elif div_gate == GateOutcome.WARN.value:
        p1.append(
            "scenario_diversity_warn:"
            f"unique={unique},concentration={conc}%,repetitive={report.diversity.get('repetitive_families')}"
        )

    if cont_bal != GateOutcome.PASS.value:
        p1.append(
            f"continuity_family_volume_low:state={state_n},decision_context={dc_n},loop="
            f"{per_family.get('loop_decision', 0)}"
        )

    amb = int(report.odoo.get("ambiguous") or 0)
    if amb >= 20:
        p1.append(f"ambiguous_odoo_generic_labels_high:{amb}")
    elif amb > 0:
        p2.append(f"ambiguous_odoo_generic_labels:{amb}")
    if domain_gate == GateOutcome.FAIL.value:
        p1.append(
            f"domain_balance_fail:odoo_pct={odoo_pct},ambiguous={amb}"
        )

    if under:
        p2.append(f"underrepresented_capabilities:{','.join(under)}")
    if over:
        p2.append(f"overrepresented_capabilities:{','.join(over)}")
    if train_only:
        p2.append(f"capabilities_train_only:{','.join(train_only)}")
    if int(report.objective_completion.get("repair_then_validate_plan_pairs") or 0) > 5:
        p2.append(
            "repair_then_validate_plan_bias:"
            f"{report.objective_completion.get('repair_then_validate_plan_pairs')}"
        )
    p3.append("pack_prompt_is_generic_contract_preamble_could_be_task_specific")
    p3.append("context_provider_pack_underrepresented_in_development_formatter_outputs")

    report.issues = {"P0": p0, "P1": p1, "P2": p2, "P3": p3}

    # Readiness decision
    rationale: list[str] = []
    if p0:
        readiness = TrainingReadiness.NOT_READY
        rationale.extend(p0)
    elif p1:
        readiness = TrainingReadiness.READY_WITH_REQUIRED_DATA_FIXES
        rationale.extend(p1)
        rationale.append("structural_gates_pass_but_semantic_pack_quality_requires_fixes")
    else:
        readiness = TrainingReadiness.READY_FOR_TRAINING
        rationale.append("hard_gates_pass_and_soft_metrics_acceptable")

    report.readiness = readiness.value
    report.readiness_rationale = rationale
    return report


def write_tr6_report(report: Tr6Report, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
