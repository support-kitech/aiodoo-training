"""EngineeringDecisionContext FP2-native generator."""

from __future__ import annotations

from typing import Any

from aiodoo_training.system_training_contract.generators.common import fixture_metadata, rid
from aiodoo_training.system_training_contract.records import DecisionContextRecord

_SPECS: tuple[dict[str, Any], ...] = (
    {
        "objective": "Add partner field and validate",
        "objective_state": "incomplete",
        "cycle_index": 1,
        "execution_state": "failed",
        "observation_quality": "failed",
        "validation_status": "failed",
        "repair_status": "not_started",
        "expected_outcomes": {"validation_ok": True},
        "missing_outcomes": ("validation",),
        "failures": ("validation_failed",),
        "possible_next_actions": ("replan", "correct", "escalate"),
        "continuation_hint": "replan",
        "bounded_history": (),
        "domain": "odoo",
        "provider": "planner",
    },
    {
        "objective": "Add partner field and validate",
        "objective_state": "in_progress",
        "cycle_index": 2,
        "execution_state": "succeeded",
        "observation_quality": "succeeded",
        "validation_status": "pending",
        "repair_status": "applied",
        "expected_outcomes": {"validation_ok": True},
        "missing_outcomes": ("validation",),
        "possible_next_actions": ("continue", "replan"),
        "continuation_hint": "continue",
        "bounded_history": (
            {"cycle_index": 1, "objective_state": "incomplete", "note": "validation_failed"},
        ),
        "domain": "odoo",
        "provider": "planner",
    },
    {
        "objective": "Add partner field and validate",
        "objective_state": "complete",
        "cycle_index": 3,
        "execution_state": "succeeded",
        "observation_quality": "succeeded",
        "validation_status": "passed",
        "repair_status": "applied",
        "expected_outcomes": {"validation_ok": True},
        "missing_outcomes": (),
        "possible_next_actions": ("complete",),
        "continuation_hint": "complete",
        "bounded_history": (
            {"cycle_index": 1, "note": "validation_failed"},
            {"cycle_index": 2, "note": "repair_applied"},
        ),
        "domain": "odoo",
        "provider": "planner",
    },
    {
        "objective": "Clarify ambiguous requirement",
        "objective_state": "blocked",
        "cycle_index": 1,
        "blockers": ("ambiguous_requirement",),
        "possible_next_actions": ("clarify", "escalate"),
        "continuation_hint": "clarify",
        "provider": "conversation",
        "domain": None,
    },
    {
        "objective": "Approve published artifact",
        "objective_state": "blocked",
        "cycle_index": 2,
        "blockers": ("awaiting_approval",),
        "possible_next_actions": ("approve", "reject", "modify"),
        "continuation_hint": "approve",
        "provider": "approval",
        "domain": None,
    },
    {
        "objective": "Recover after timed out execution",
        "objective_state": "incomplete",
        "cycle_index": 2,
        "execution_state": "timed_out",
        "failures": ("timed_out",),
        "possible_next_actions": ("retry", "recover", "replan"),
        "continuation_hint": "recover",
        "provider": "planner",
        "domain": None,
    },
    {
        "objective": "Evaluate candidate outcome",
        "objective_state": "in_progress",
        "cycle_index": 1,
        "observation_quality": "succeeded",
        "possible_next_actions": ("continue", "complete"),
        "continuation_hint": "continue",
        "provider": "evaluation",
        "domain": None,
    },
    {
        "objective": "Remote dependency missing",
        "objective_state": "escalated",
        "cycle_index": 2,
        "blockers": ("remote_unreachable",),
        "failures": ("http_failed",),
        "possible_next_actions": ("escalate", "cancel"),
        "continuation_hint": "escalate",
        "provider": "planner",
        "domain": None,
    },
    {
        "objective": "Prior complete must not dominate current failure",
        "objective_state": "failed",
        "cycle_index": 5,
        "validation_status": "failed",
        "failures": ("validation_failed",),
        "bounded_history": (
            {"cycle_index": 4, "objective_state": "complete", "historical": True},
        ),
        "possible_next_actions": ("replan", "escalate"),
        "continuation_hint": "replan",
        "provider": "planner",
        "domain": None,
    },
    {
        "objective": "Prior failure must not dominate current success",
        "objective_state": "complete",
        "cycle_index": 2,
        "validation_status": "passed",
        "bounded_history": (
            {"cycle_index": 1, "objective_state": "failed", "historical": True},
        ),
        "possible_next_actions": ("complete",),
        "continuation_hint": "complete",
        "provider": "planner",
        "domain": "odoo",
    },
    {
        "objective": "Missing outcomes only",
        "objective_state": "incomplete",
        "cycle_index": 1,
        "execution_state": "succeeded",
        "missing_outcomes": ("artifact", "validation"),
        "possible_next_actions": ("continue",),
        "continuation_hint": "continue",
        "provider": "execution",
        "domain": None,
    },
    {
        "objective": "No blockers — ready to complete",
        "objective_state": "complete",
        "cycle_index": 1,
        "execution_state": "succeeded",
        "observation_quality": "succeeded",
        "validation_status": "passed",
        "possible_next_actions": ("complete",),
        "continuation_hint": "complete",
        "provider": "planner",
        "domain": None,
    },
)


def generate_decision_contexts() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, spec in enumerate(_SPECS, start=1):
        provider = str(spec.get("provider") or "planner")
        # approval is Reasoning but DecisionContextRecord.provider_capability must be
        # a provider pack — approval is valid. conversation/evaluation too.
        if provider == "approval":
            # DecisionContext teaches planner/conversation/evaluation primarily;
            # approval decisions use loop_decision. Keep planner label for schema.
            provider_field = "planner"
        else:
            provider_field = provider
        domain = spec.get("domain")
        rec = DecisionContextRecord(
            record_type="decision_context",
            record_id=rid("dc", i),
            objective=str(spec["objective"]),
            objective_state=str(spec["objective_state"]),
            cycle_index=int(spec["cycle_index"]),
            execution_state=str(spec.get("execution_state") or ""),
            observation_quality=str(spec.get("observation_quality") or ""),
            validation_status=str(spec.get("validation_status") or ""),
            repair_status=str(spec.get("repair_status") or ""),
            expected_outcomes=dict(spec.get("expected_outcomes") or {}),
            missing_outcomes=tuple(spec.get("missing_outcomes") or ()),
            blockers=tuple(spec.get("blockers") or ()),
            failures=tuple(spec.get("failures") or ()),
            possible_next_actions=tuple(spec.get("possible_next_actions") or ()),
            continuation_hint=str(spec.get("continuation_hint") or ""),
            bounded_history=tuple(spec.get("bounded_history") or ()),
            provider_capability=provider_field,
            domain_specialization=domain if isinstance(domain, str) else None,
            metadata=fixture_metadata(
                generator="decision_context",
                index=i,
                provider_capability=provider_field,
                domain_specialization=domain if isinstance(domain, str) else None,
                extra={"intended_consumer": provider},
            ),
        )
        out.append(rec.to_dict())
    return out
