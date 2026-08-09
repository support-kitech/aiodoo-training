"""EngineeringState FP2-native generator — current-cycle isolation fixtures."""

from __future__ import annotations

from typing import Any

from aiodoo_training.system_training_contract.generators.common import fixture_metadata, rid
from aiodoo_training.system_training_contract.records import EngineeringStateRecord

# Explicit multi-cycle narrative for the same objective (no historical overwrite).
_SPECS: tuple[dict[str, Any], ...] = (
    {
        "objective": "Add partner field and validate",
        "objective_state": "incomplete",
        "session_state": "active",
        "completion_state": "open",
        "cycle_index": 1,
        "current_fields": {"validation_status": "failed", "repair_status": "not_started"},
        "scenario": "cycle1_validation_failed",
        "domain": "odoo",
    },
    {
        "objective": "Add partner field and validate",
        "objective_state": "in_progress",
        "session_state": "active",
        "completion_state": "open",
        "cycle_index": 2,
        "current_fields": {"validation_status": "pending", "repair_status": "applied"},
        "scenario": "cycle2_repair_applied",
        "domain": "odoo",
    },
    {
        "objective": "Add partner field and validate",
        "objective_state": "complete",
        "session_state": "active",
        "completion_state": "ready",
        "cycle_index": 3,
        "current_fields": {"validation_status": "passed", "repair_status": "applied"},
        "scenario": "cycle3_validation_passed",
        "domain": "odoo",
    },
    {
        "objective": "Ship hotfix",
        "objective_state": "complete",
        "session_state": "idle",
        "completion_state": "complete",
        "cycle_index": 4,
        "current_fields": {"prior_outcome": "complete"},
        "scenario": "previous_complete",
        "domain": None,
    },
    {
        "objective": "Ship hotfix",
        "objective_state": "failed",
        "session_state": "active",
        "completion_state": "open",
        "cycle_index": 5,
        "current_fields": {
            "validation_status": "failed",
            "historical_summary_note": "prior_complete_must_not_overwrite",
        },
        "scenario": "previous_complete_current_failure",
        "domain": None,
    },
    {
        "objective": "Recover module",
        "objective_state": "failed",
        "session_state": "active",
        "completion_state": "open",
        "cycle_index": 1,
        "current_fields": {"validation_status": "failed"},
        "scenario": "previous_failure",
        "domain": "odoo",
    },
    {
        "objective": "Recover module",
        "objective_state": "complete",
        "session_state": "active",
        "completion_state": "ready",
        "cycle_index": 2,
        "current_fields": {
            "validation_status": "passed",
            "historical_summary_note": "prior_failure_must_not_overwrite_current_success",
        },
        "scenario": "previous_failure_current_success",
        "domain": "odoo",
    },
    {
        "objective": "Inspect only",
        "objective_state": "in_progress",
        "session_state": "active",
        "completion_state": "open",
        "cycle_index": 0,
        "current_fields": {"capability_id": "repository.inspect"},
        "scenario": "inspect_cycle0",
        "domain": None,
    },
    {
        "objective": "Blocked on approval",
        "objective_state": "blocked",
        "session_state": "waiting",
        "completion_state": "open",
        "cycle_index": 2,
        "current_fields": {"approval_state": "required"},
        "scenario": "blocked_approval",
        "domain": None,
    },
    {
        "objective": "Escalated outage",
        "objective_state": "escalated",
        "session_state": "escalated",
        "completion_state": "open",
        "cycle_index": 3,
        "current_fields": {"blockers": ["remote_unreachable"]},
        "scenario": "escalated",
        "domain": None,
    },
    {
        "objective": "Cancelled session",
        "objective_state": "cancelled",
        "session_state": "cancelled",
        "completion_state": "cancelled",
        "cycle_index": 1,
        "current_fields": {},
        "scenario": "cancelled",
        "domain": None,
    },
    {
        "objective": "Unknown early state",
        "objective_state": "unknown",
        "session_state": "active",
        "completion_state": "open",
        "cycle_index": 0,
        "current_fields": {},
        "scenario": "unknown",
        "domain": None,
    },
)


def generate_states() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, spec in enumerate(_SPECS, start=1):
        domain = spec.get("domain")
        rec = EngineeringStateRecord(
            record_type="engineering_state",
            record_id=rid("st", i),
            objective=str(spec["objective"]),
            objective_state=str(spec["objective_state"]),
            session_state=str(spec["session_state"]),
            completion_state=str(spec["completion_state"]),
            cycle_index=int(spec["cycle_index"]),
            current_fields=dict(spec.get("current_fields") or {}),
            provider_capability="planner",
            domain_specialization=domain if isinstance(domain, str) else None,
            metadata=fixture_metadata(
                generator="engineering_state",
                index=i,
                provider_capability="planner",
                domain_specialization=domain if isinstance(domain, str) else None,
                extra={"scenario": str(spec["scenario"])},
            ),
        )
        out.append(rec.to_dict())
    return out
