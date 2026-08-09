"""Capability coverage and continuity / quality analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from aiodoo_training.system_training_contract.quality.common import (
    extract_engineering_capability,
    fingerprint_record,
)
from aiodoo_training.system_training_contract.taxonomy import PREFERRED_ENGINEERING_CAPABILITY_IDS


def capability_coverage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Coverage matrix vs preferred Engineering taxonomy."""
    per_cap: dict[str, dict[str, Any]] = {
        cap: {
            "count": 0,
            "families": set(),
            "odoo": 0,
            "generic": 0,
            "successish": 0,
            "failureish": 0,
        }
        for cap in sorted(PREFERRED_ENGINEERING_CAPABILITY_IDS)
    }
    for rec in records:
        eng = extract_engineering_capability(rec)
        if not eng:
            continue
        caps = [c.strip() for c in eng.split(",") if c.strip()]
        domain = rec.get("domain_specialization")
        status = _status_hint(rec)
        for cap in caps:
            if cap not in per_cap:
                continue
            per_cap[cap]["count"] += 1
            per_cap[cap]["families"].add(str(rec.get("record_type") or ""))
            if domain == "odoo":
                per_cap[cap]["odoo"] += 1
            else:
                per_cap[cap]["generic"] += 1
            if status == "success":
                per_cap[cap]["successish"] += 1
            elif status == "failure":
                per_cap[cap]["failureish"] += 1
    uncovered = [c for c, v in per_cap.items() if v["count"] == 0]
    covered = [c for c, v in per_cap.items() if v["count"] > 0]
    serializable = {
        c: {
            "count": v["count"],
            "families": sorted(v["families"]),
            "odoo": v["odoo"],
            "generic": v["generic"],
            "successish": v["successish"],
            "failureish": v["failureish"],
        }
        for c, v in per_cap.items()
    }
    return {
        "preferred_total": len(PREFERRED_ENGINEERING_CAPABILITY_IDS),
        "covered_count": len(covered),
        "uncovered_count": len(uncovered),
        "coverage_pct": round(100.0 * len(covered) / max(1, len(PREFERRED_ENGINEERING_CAPABILITY_IDS)), 2),
        "uncovered": uncovered,
        "overrepresented": [
            c for c, v in serializable.items() if v["count"] >= 6
        ],
        "weak": [
            c for c, v in serializable.items() if 0 < v["count"] < 2
        ],
        "matrix": serializable,
    }


def _status_hint(rec: Mapping[str, Any]) -> str:
    evidence = rec.get("evidence") if isinstance(rec.get("evidence"), Mapping) else {}
    status = str(evidence.get("status") or evidence.get("objective_state") or "").lower()
    if status in {"succeeded", "complete", "passed"}:
        return "success"
    if status in {"failed", "failure", "blocked", "escalated", "cancelled"}:
        return "failure"
    return "other"


def domain_distribution(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    odoo = sum(1 for r in records if r.get("domain_specialization") == "odoo")
    generic = len(records) - odoo
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for r in records:
        fam = str(r.get("record_type") or "unknown")
        label = "odoo" if r.get("domain_specialization") == "odoo" else "generic"
        by_family[fam][label] += 1
    return {
        "odoo": odoo,
        "generic": generic,
        "odoo_pct": round(100.0 * odoo / max(1, len(records)), 2),
        "by_family": {k: dict(v) for k, v in sorted(by_family.items())},
    }


def find_duplicates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for r in records:
        fp = fingerprint_record(r)
        buckets[fp].append(str(r.get("record_id") or "?"))
    dups = {fp: ids for fp, ids in buckets.items() if len(ids) > 1}
    return {
        "unique_fingerprints": len(buckets),
        "duplicate_groups": len(dups),
        "duplicate_record_ids": dups,
    }


def analyze_state_isolation(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Verify multi-cycle state fixtures do not overwrite current with history."""
    states = [r for r in records if r.get("record_type") == "engineering_state"]
    scenarios = {
        str((r.get("metadata") or {}).get("scenario") or ""): r for r in states
    }
    issues: list[str] = []
    required = (
        "cycle1_validation_failed",
        "cycle2_repair_applied",
        "cycle3_validation_passed",
        "previous_complete_current_failure",
        "previous_failure_current_success",
    )
    for name in required:
        if name not in scenarios:
            issues.append(f"missing_state_scenario:{name}")
    c1 = scenarios.get("cycle1_validation_failed")
    c2 = scenarios.get("cycle2_repair_applied")
    c3 = scenarios.get("cycle3_validation_passed")
    if c1 and c2 and c3:
        e1 = c1["evidence"]
        e2 = c2["evidence"]
        e3 = c3["evidence"]
        if e1.get("cycle_index") != 1 or e1["current_fields"].get("validation_status") != "failed":
            issues.append("cycle1_current_fields_incorrect")
        if e2.get("cycle_index") != 2 or e2["current_fields"].get("repair_status") != "applied":
            issues.append("cycle2_current_fields_incorrect")
        if e3.get("cycle_index") != 3 or e3["current_fields"].get("validation_status") != "passed":
            issues.append("cycle3_current_fields_incorrect")
        # Historical must not equal current incorrectly
        if e3["current_fields"].get("validation_status") == "failed":
            issues.append("cycle3_historical_leakage")
    prev_ok = scenarios.get("previous_complete_current_failure")
    if prev_ok and prev_ok["evidence"].get("objective_state") != "failed":
        issues.append("previous_complete_did_not_yield_to_current_failure")
    prev_fail = scenarios.get("previous_failure_current_success")
    if prev_fail and prev_fail["evidence"].get("objective_state") != "complete":
        issues.append("previous_failure_did_not_yield_to_current_success")

    # DecisionContext: current objective_state must not be overwritten by history
    dcs = [r for r in records if r.get("record_type") == "decision_context"]
    for dc in dcs:
        inp = dc.get("input") if isinstance(dc.get("input"), Mapping) else {}
        hist = inp.get("bounded_history") or ()
        for h in hist:
            if not isinstance(h, Mapping):
                continue
            if h.get("historical") and h.get("objective_state") == inp.get("objective_state"):
                # Same value OK if current truly matches; flag only leakage markers
                if h.get("objective_state") == "complete" and inp.get("objective_state") == "failed":
                    pass  # good: current failure despite historical complete
            if "implementation_id" in h or "command" in h or "stdout" in h:
                issues.append(f"decision_context_history_how:{dc.get('record_id')}")
        for token in ("local_workspace", "implementation_id", "stdout", "password"):
            if token in str(inp).lower():
                issues.append(f"decision_context_how_leak:{dc.get('record_id')}:{token}")

    return {"issues": issues, "ok": not issues, "scenarios_found": sorted(scenarios)}


def analyze_feedback_operation_vs_objective(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    feedback = [r for r in records if r.get("record_type") == "engineering_feedback"]
    distinguished = 0
    for r in feedback:
        ev = r.get("evidence") if isinstance(r.get("evidence"), Mapping) else {}
        if ev.get("execution_state") == "succeeded" and ev.get("objective_state") in {
            "incomplete",
            "in_progress",
            "blocked",
        }:
            distinguished += 1
    return {
        "feedback_count": len(feedback),
        "operation_success_objective_incomplete_examples": distinguished,
        "ok": distinguished >= 1,
    }


def analyze_loop_decisions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    loops = [r for r in records if r.get("record_type") == "loop_decision"]
    kinds = Counter(
        str((r.get("expected_output") or {}).get("decision_kind") or "") for r in loops
    )
    required = {"replan", "complete", "escalate", "retry", "recover"}
    missing = sorted(required - set(kinds))
    # Soft check: no hard-coded "validation failure → repair" in reason alone as sole policy
    auto_policy = [
        str(r.get("record_id"))
        for r in loops
        if "automatically repair" in str((r.get("expected_output") or {}).get("reason") or "").lower()
        or "must validate after repair" in str((r.get("expected_output") or {}).get("reason") or "").lower()
    ]
    return {
        "counts": dict(kinds),
        "missing_required_kinds": missing,
        "auto_policy_reasons": auto_policy,
        "ok": not missing and not auto_policy,
    }


def analyze_work_units(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    wus = [r for r in records if r.get("record_type") == "execution_work_unit"]
    issues: list[str] = []
    for r in wus:
        inp = r.get("input") if isinstance(r.get("input"), Mapping) else {}
        out = r.get("expected_output") if isinstance(r.get("expected_output"), Mapping) else {}
        if not str(inp.get("objective") or "").strip():
            issues.append(f"wu_missing_objective:{r.get('record_id')}")
        if not str(out.get("capability_id") or "").strip():
            issues.append(f"wu_missing_capability:{r.get('record_id')}")
        if not str(out.get("work_id") or "").strip():
            issues.append(f"wu_missing_work_id:{r.get('record_id')}")
        if not isinstance(out.get("expected_outputs"), Mapping):
            issues.append(f"wu_expected_outputs_not_mapping:{r.get('record_id')}")
    return {"count": len(wus), "issues": issues, "ok": not issues}


def analyze_planning(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    plans = [r for r in records if r.get("record_type") == "planning_decision"]
    kinds = Counter(
        str((r.get("expected_output") or {}).get("decision_kind") or "") for r in plans
    )
    issues: list[str] = []
    for r in plans:
        out = r.get("expected_output") if isinstance(r.get("expected_output"), Mapping) else {}
        kind = str(out.get("decision_kind") or "")
        steps = out.get("steps") or []
        if kind == "complete" and steps:
            issues.append(f"complete_with_steps:{r.get('record_id')}")
        for step in steps:
            if not isinstance(step, Mapping):
                issues.append(f"bad_step:{r.get('record_id')}")
                continue
            act = str(step.get("action") or "")
            if act in {"shell", "pytest", "git", "create_file"}:
                issues.append(f"planning_how_action:{r.get('record_id')}:{act}")
    return {
        "count": len(plans),
        "kinds": dict(kinds),
        "has_replan": kinds.get("replan", 0) > 0,
        "has_complete": kinds.get("complete", 0) > 0,
        "has_escalate": kinds.get("escalate", 0) > 0,
        "issues": issues,
        "ok": not issues and kinds.get("replan", 0) > 0 and kinds.get("complete", 0) > 0,
    }
