"""Adapter / record-type mapping for FP2-native corpora (no adapter chaining)."""

from __future__ import annotations

from aiodoo_training.system_training_contract.taxonomy import (
    DEVELOPMENT_PROVIDER_CAPABILITIES,
    REASONING_PROVIDER_CAPABILITIES,
)

# Which provider packs may consume each FP2 record family (shared, independent).
#
# Context (AT-6.2): Development retrieval / repository-locate specialization.
# Allowed on locate/search families only — NOT decision_context (Decision Continuity),
# NOT execution_work_unit (TRAINING_SYSTEM_CONTRACT: context → preserve; not Work Units),
# NOT Reasoning loop/planning families. See docs/AT6_2_CONTEXT_GENERATOR.md.
#
# Evaluation (AT-7.4): ONLY evaluation_judgment (EvaluationRequest/Response surface).
# Removed from observation / engineering_feedback / decision_context — those families
# do not encode judgment semantics (AT-7.3 block). See docs/AT7_4_EVALUATION_CONTRACT.md.
_RECORD_TO_PROVIDERS: dict[str, frozenset[str]] = {
    "capability_intent": frozenset({"planner", "execution", "repair", "coding", "context"}),
    "execution_work_unit": frozenset({"execution", "planner", "repair"}),
    "planning_decision": frozenset({"planner"}),
    "observation": frozenset({"execution", "repair", "context"}),
    "engineering_feedback": frozenset({"planner", "execution"}),
    "engineering_state": frozenset({"planner", "execution"}),
    "decision_context": frozenset({"planner", "conversation"}),
    "loop_decision": frozenset({"planner", "approval", "conversation"}),
    "evaluation_judgment": frozenset({"evaluation"}),
}

DEVELOPMENT_RECORD_TYPES: frozenset[str] = frozenset(
    {
        "capability_intent",
        "execution_work_unit",
        "observation",
        "engineering_feedback",
        "engineering_state",
    }
)

REASONING_RECORD_TYPES: frozenset[str] = frozenset(
    {
        "capability_intent",
        "planning_decision",
        "engineering_feedback",
        "decision_context",
        "loop_decision",
        "observation",
        "evaluation_judgment",
    }
)

# Record types that may carry provider_capability=context (AT-6.2).
CONTEXT_ALLOWED_RECORD_TYPES: frozenset[str] = frozenset(
    {
        "capability_intent",
        "observation",
    }
)

CONTEXT_REJECTED_RECORD_TYPES: frozenset[str] = frozenset(
    {
        "execution_work_unit",
        "decision_context",
        "planning_decision",
        "loop_decision",
        "engineering_state",
        "engineering_feedback",
        "evaluation_judgment",
    }
)

# Record types that may carry provider_capability=evaluation (AT-7.4).
EVALUATION_ALLOWED_RECORD_TYPES: frozenset[str] = frozenset({"evaluation_judgment"})

EVALUATION_REJECTED_RECORD_TYPES: frozenset[str] = frozenset(
    {
        "capability_intent",
        "execution_work_unit",
        "planning_decision",
        "observation",
        "engineering_feedback",
        "engineering_state",
        "decision_context",
        "loop_decision",
    }
)


def record_provider_capabilities(record_type: str) -> frozenset[str]:
    return _RECORD_TO_PROVIDERS.get(record_type, frozenset())


def assert_no_adapter_chain() -> None:
    # Independence invariant: Development and Reasoning share packs only via
    # shared record types — neither set is a prerequisite of the other.
    assert DEVELOPMENT_PROVIDER_CAPABILITIES.isdisjoint(REASONING_PROVIDER_CAPABILITIES)
    # Context is Development-only and must never appear on Reasoning-only families.
    for rtype in CONTEXT_REJECTED_RECORD_TYPES:
        assert "context" not in record_provider_capabilities(rtype)
    for rtype in CONTEXT_ALLOWED_RECORD_TYPES:
        assert "context" in record_provider_capabilities(rtype)
    # Evaluation is Reasoning-only and must only appear on evaluation_judgment.
    for rtype in EVALUATION_REJECTED_RECORD_TYPES:
        assert "evaluation" not in record_provider_capabilities(rtype)
    for rtype in EVALUATION_ALLOWED_RECORD_TYPES:
        assert record_provider_capabilities(rtype) == frozenset({"evaluation"})
    assert "evaluation_judgment" in REASONING_RECORD_TYPES
    assert "evaluation_judgment" not in DEVELOPMENT_RECORD_TYPES
