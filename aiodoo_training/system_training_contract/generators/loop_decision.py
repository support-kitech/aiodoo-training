"""Loop Decision FP2-native generator — evidence-based, no hard-coded repair policy."""

from __future__ import annotations

from typing import Any

from aiodoo_training.system_training_contract.generators.common import fixture_metadata, rid
from aiodoo_training.system_training_contract.records import LoopDecisionRecord

_SPECS: tuple[dict[str, Any], ...] = (
    {
        "decision_kind": "replan",
        "reason": "Missing validation outcome after successful write",
        "next_goal": "Plan validation.run for current objective",
        "provider": "planner",
        "domain": "odoo",
    },
    {
        "decision_kind": "complete",
        "reason": "Objective complete; validation passed; no blockers",
        "next_goal": "",
        "provider": "planner",
        "domain": None,
    },
    {
        "decision_kind": "escalate",
        "reason": "Remote dependency unreachable and retries exhausted",
        "next_goal": "",
        "provider": "planner",
        "domain": None,
    },
    {
        "decision_kind": "retry",
        "reason": "Transient execution failure with unchanged objective",
        "next_goal": "Retry last Engineering capability intent",
        "provider": "planner",
        "domain": None,
    },
    {
        "decision_kind": "recover",
        "reason": "Session recoverable after timed_out execution",
        "next_goal": "Recover and re-observe",
        "provider": "planner",
        "domain": None,
    },
    {
        "decision_kind": "continue",
        "reason": "Partial progress; remaining missing outcomes",
        "next_goal": "Continue current plan",
        "provider": "planner",
        "domain": "odoo",
    },
    {
        "decision_kind": "clarify",
        "reason": "Requirement ambiguous; need user clarification",
        "next_goal": "Ask clarifying questions",
        "provider": "conversation",
        "domain": None,
    },
    {
        "decision_kind": "approve",
        "reason": "Human approval required before publish",
        "next_goal": "Wait for approval",
        "provider": "approval",
        "domain": None,
    },
    {
        "decision_kind": "reject",
        "reason": "Reviewer rejects proposed change set",
        "next_goal": "Return to planning with rejection notes",
        "provider": "approval",
        "domain": None,
    },
    {
        "decision_kind": "modify",
        "reason": "Approval requests modification before accept",
        "next_goal": "Replan with reviewer constraints",
        "provider": "approval",
        "domain": None,
    },
    {
        "decision_kind": "cancel",
        "reason": "Operator cancelled the engineering session",
        "next_goal": "",
        "provider": "planner",
        "domain": None,
    },
    {
        "decision_kind": "pause",
        "reason": "Pause until external dependency window opens",
        "next_goal": "",
        "provider": "planner",
        "domain": None,
    },
)


def generate_loop_decisions() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, spec in enumerate(_SPECS, start=1):
        provider = str(spec["provider"])
        domain = spec.get("domain")
        rec = LoopDecisionRecord(
            record_type="loop_decision",
            record_id=rid("ld", i),
            decision_kind=str(spec["decision_kind"]),
            reason=str(spec["reason"]),
            next_goal=str(spec.get("next_goal") or ""),
            provider_capability=provider,
            domain_specialization=domain if isinstance(domain, str) else None,
            metadata=fixture_metadata(
                generator="loop_decision",
                index=i,
                provider_capability=provider,
                domain_specialization=domain if isinstance(domain, str) else None,
            ),
        )
        out.append(rec.to_dict())
    return out
