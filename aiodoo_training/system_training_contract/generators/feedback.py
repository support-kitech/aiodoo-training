"""EngineeringFeedback FP2-native generator."""

from __future__ import annotations

from typing import Any

from aiodoo_training.system_training_contract.generators.common import fixture_metadata, rid
from aiodoo_training.system_training_contract.records import EngineeringFeedbackRecord

_SPECS: tuple[dict[str, Any], ...] = (
    {
        "objective": "Add partner field",
        "objective_state": "incomplete",
        "execution_state": "succeeded",
        "observation_quality": "succeeded",
        "note": "operation succeeded but objective incomplete",
        "continuation_options": ("continue", "complete"),
        "recommended_continuation": "continue",
        "missing_outcomes": ("validation",),
        "provider": "planner",
        "domain": "odoo",
    },
    {
        "objective": "Add partner field",
        "objective_state": "incomplete",
        "execution_state": "failed",
        "observation_quality": "failed",
        "validation": {"ok": False, "failed_checks": ["imports"]},
        "failures": ("validation_failed",),
        "continuation_options": ("correct", "replan", "escalate"),
        "recommended_continuation": "correct",
        "provider": "execution",
        "domain": "odoo",
    },
    {
        "objective": "Repair import",
        "objective_state": "failed",
        "execution_state": "failed",
        "observation_quality": "failed",
        "failures": ("repair_failed",),
        "continuation_options": ("retry", "recover", "escalate"),
        "recommended_continuation": "recover",
        "provider": "repair",
        "domain": "odoo",
    },
    {
        "objective": "Ship change",
        "objective_state": "incomplete",
        "execution_state": "partial",
        "observation_quality": "partial",
        "missing_outcomes": ("expected_artifact",),
        "continuation_options": ("continue", "correct"),
        "recommended_continuation": "continue",
        "provider": "execution",
        "domain": None,
    },
    {
        "objective": "Ship change",
        "objective_state": "blocked",
        "execution_state": "succeeded",
        "observation_quality": "succeeded",
        "blockers": ("missing_approval",),
        "continuation_options": ("wait", "escalate"),
        "recommended_continuation": "wait",
        "provider": "planner",
        "domain": None,
    },
    {
        "objective": "Ship change",
        "objective_state": "complete",
        "execution_state": "succeeded",
        "observation_quality": "succeeded",
        "validation": {"ok": True},
        "continuation_options": ("complete",),
        "recommended_continuation": "complete",
        "provider": "planner",
        "domain": None,
    },
    {
        "objective": "Validate module",
        "objective_state": "in_progress",
        "execution_state": "missing_observation",
        "observation_quality": "missing",
        "continuation_options": ("retry", "escalate"),
        "recommended_continuation": "retry",
        "provider": "execution",
        "domain": "odoo",
    },
    {
        "objective": "Apply repair then continue",
        "objective_state": "in_progress",
        "execution_state": "succeeded",
        "observation_quality": "succeeded",
        "expected_outputs": {"import_ok": True},
        "actual_outputs": {"import_ok": True},
        "missing_outcomes": ("validation",),
        "continuation_options": ("continue",),
        "recommended_continuation": "continue",
        "provider": "repair",
        "domain": "odoo",
    },
    {
        "objective": "External dependency unavailable",
        "objective_state": "escalated",
        "execution_state": "failed",
        "observation_quality": "failed",
        "blockers": ("remote_unreachable"),
        "continuation_options": ("escalate", "cancel"),
        "recommended_continuation": "escalate",
        "provider": "planner",
        "domain": None,
    },
    {
        "objective": "Cancelled by operator",
        "objective_state": "cancelled",
        "execution_state": "cancelled",
        "observation_quality": "partial",
        "continuation_options": ("cancel",),
        "recommended_continuation": "cancel",
        "provider": "planner",
        "domain": None,
    },
    {
        "objective": "Timed out program execution",
        "objective_state": "incomplete",
        "execution_state": "timed_out",
        "observation_quality": "incomplete",
        "failures": ("timed_out",),
        "continuation_options": ("retry", "replan"),
        "recommended_continuation": "retry",
        "provider": "execution",
        "domain": None,
    },
    {
        "objective": "Another cycle required after partial write",
        "objective_state": "incomplete",
        "execution_state": "succeeded",
        "observation_quality": "succeeded",
        "missing_outcomes": ("validation", "artifact"),
        "continuation_options": ("continue", "correct"),
        "recommended_continuation": "continue",
        "provider": "evaluation",
        "domain": "odoo",
    },
)


def generate_feedback() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, spec in enumerate(_SPECS, start=1):
        provider = str(spec["provider"])
        domain = spec.get("domain")
        blockers = spec.get("blockers") or ()
        if isinstance(blockers, str):
            blockers = (blockers,)
        rec = EngineeringFeedbackRecord(
            record_type="engineering_feedback",
            record_id=rid("fb", i),
            objective=str(spec["objective"]),
            objective_state=str(spec["objective_state"]),
            execution_state=str(spec["execution_state"]),
            observation_quality=str(spec["observation_quality"]),
            continuation_options=tuple(spec.get("continuation_options") or ()),
            recommended_continuation=str(spec.get("recommended_continuation") or ""),
            observations=tuple(spec.get("observations") or ()),
            blockers=tuple(blockers),
            failures=tuple(spec.get("failures") or ()),
            missing_outcomes=tuple(spec.get("missing_outcomes") or ()),
            validation=dict(spec.get("validation") or {}),
            expected_outputs=dict(spec.get("expected_outputs") or {}),
            actual_outputs=dict(spec.get("actual_outputs") or {}),
            provider_capability=provider,
            domain_specialization=domain if isinstance(domain, str) else None,
            metadata=fixture_metadata(
                generator="engineering_feedback",
                index=i,
                provider_capability=provider,
                domain_specialization=domain if isinstance(domain, str) else None,
                extra={"scenario": str(spec.get("note") or spec["objective_state"])},
            ),
        )
        out.append(rec.to_dict())
    return out
