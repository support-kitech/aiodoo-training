"""Negative / adversarial quality fixtures (NOT training data)."""

from __future__ import annotations

from typing import Any

# Each case: (case_id, expected_accepted: bool, record_or_projection_payload)
# These must never be mixed into production training output.
NEGATIVE_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "neg_forbidden_local",
        "expected": "rejected",
        "reason": "forbidden HOW local_workspace as capability",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "capability_intent",
            "record_id": "neg-001",
            "system_contract": "execution.capability_intent",
            "provider_capability": "planner",
            "input": {"objective": "do work", "reason": "x"},
            "expected_output": {"capability_id": "local_workspace", "args": {}},
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_invalid_capability",
        "expected": "rejected",
        "reason": "unknown Engineering capability",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "capability_intent",
            "record_id": "neg-002",
            "system_contract": "execution.capability_intent",
            "provider_capability": "planner",
            "input": {"objective": "do work", "reason": "x"},
            "expected_output": {"capability_id": "fabricated.magic", "args": {}},
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_provider_as_engineering",
        "expected": "rejected",
        "reason": "provider pack used as Engineering action",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "capability_intent",
            "record_id": "neg-003",
            "system_contract": "execution.capability_intent",
            "provider_capability": "planner",
            "input": {"objective": "code", "reason": "x"},
            "expected_output": {"capability_id": "coding", "args": {}},
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_engineering_as_provider",
        "expected": "rejected",
        "reason": "Engineering ID used as provider_capability",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "capability_intent",
            "record_id": "neg-004",
            "system_contract": "execution.capability_intent",
            "provider_capability": "workspace.write",
            "input": {"objective": "write", "reason": "x"},
            "expected_output": {"capability_id": "workspace.write", "args": {"path": "a.py"}},
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_missing_objective",
        "expected": "rejected",
        "reason": "missing required objective",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "capability_intent",
            "record_id": "neg-005",
            "system_contract": "execution.capability_intent",
            "provider_capability": "planner",
            "input": {"objective": "", "reason": "x"},
            "expected_output": {"capability_id": "workspace.read", "args": {}},
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_wrong_record_type",
        "expected": "rejected",
        "reason": "unknown record_type",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "not_a_real_type",
            "record_id": "neg-006",
            "system_contract": "x",
            "provider_capability": "planner",
            "input": {},
            "expected_output": {},
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_invalid_loop_kind",
        "expected": "rejected",
        "reason": "invalid loop decision kind",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "loop_decision",
            "record_id": "neg-007",
            "system_contract": "intelligence_loop.decision",
            "provider_capability": "planner",
            "expected_output": {
                "decision_kind": "teleport",
                "reason": "nope",
                "next_goal": "",
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_missing_provenance_projected",
        "expected": "rejected",
        "reason": "projected record missing provenance fields",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "capability_intent",
            "record_id": "neg-008",
            "system_contract": "execution.capability_intent",
            "provider_capability": "repair",
            "input": {"objective": "repair", "reason": "projected"},
            "expected_output": {"capability_id": "execution.repair", "args": {}},
            "provenance": {},
            "metadata": {"quality_corpus": "negative", "projected": True},
        },
        "require_provenance": True,
    },
    {
        "case_id": "neg_shell_command_args",
        "expected": "rejected",
        "reason": "forbidden arg key command",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "capability_intent",
            "record_id": "neg-009",
            "system_contract": "execution.capability_intent",
            "provider_capability": "execution",
            "input": {"objective": "run", "reason": "x"},
            "expected_output": {
                "capability_id": "execution.execute_program",
                "args": {"command": "pytest -q"},
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_complete_with_steps",
        "expected": "rejected",
        "reason": "COMPLETE planning must not include steps",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "planning_decision",
            "record_id": "neg-010",
            "system_contract": "intelligence.planning",
            "provider_capability": "planner",
            "input": {"goal": "done"},
            "expected_output": {
                "decision_kind": "complete",
                "summary": "bad",
                "steps": [{"action": "workspace.read", "args": {}}],
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_contradictory_not_schema",
        "expected": "accepted_with_warn",
        "reason": "schema-valid but contradictory semantics flagged by continuity analysis elsewhere",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "engineering_state",
            "record_id": "neg-011",
            "system_contract": "execution.engineering_state",
            "provider_capability": "planner",
            "evidence": {
                "objective": "x",
                "objective_state": "complete",
                "session_state": "active",
                "completion_state": "open",
                "cycle_index": 1,
                "current_fields": {"validation_status": "failed"},
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative", "semantic_warn": "complete_with_failed_validation"},
        },
    },
    {
        "case_id": "pos_control_valid",
        "expected": "accepted",
        "reason": "control positive — valid preferred intent",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "capability_intent",
            "record_id": "pos-001",
            "system_contract": "execution.capability_intent",
            "provider_capability": "planner",
            "input": {"objective": "Read file", "reason": "control"},
            "expected_output": {"capability_id": "workspace.read", "args": {"path": "a.py"}},
            "provenance": {},
            "metadata": {"quality_corpus": "negative_control_positive"},
        },
    },
    {
        "case_id": "pos_historical_meta_ok",
        "expected": "accepted",
        "reason": "historical HOW string only in provenance/metadata, not model-facing",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "capability_intent",
            "record_id": "pos-002",
            "system_contract": "execution.capability_intent",
            "provider_capability": "planner",
            "input": {"objective": "Write file", "reason": "ok"},
            "expected_output": {"capability_id": "workspace.write", "args": {"path": "b.py"}},
            "provenance": {
                "source_dataset": "legacy",
                "source_record_id": "h1",
                "source_schema_version": "protocol-v1",
                "projection_version": "1.0.0",
                "projection_status": "projected",
                "notes": "historical create_file / local_workspace mentioned only here",
            },
            "metadata": {
                "quality_corpus": "negative_control_positive",
                "historical_note": "legacy local_git path was HOW",
            },
        },
    },
    {
        "case_id": "neg_context_forbidden_how",
        "expected": "rejected",
        "reason": "Context provider with forbidden HOW capability",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "capability_intent",
            "record_id": "neg-ctx-001",
            "system_contract": "execution.capability_intent",
            "provider_capability": "context",
            "input": {"objective": "locate model", "reason": "x"},
            "expected_output": {"capability_id": "local_workspace", "args": {}},
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_decision_context_as_context_provider_schema_ok",
        "expected": "accepted",
        "reason": "decision_context record_type with provider=context is schema-valid but policy-rejected by mapping (caught in AT-6.2 tests, not schema)",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "decision_context",
            "record_id": "neg-ctx-002",
            "system_contract": "execution.engineering_decision_context",
            "provider_capability": "context",
            "input": {
                "objective": "x",
                "objective_state": "incomplete",
                "cycle_index": 1,
                "execution_state": "failed",
                "observation_quality": "failed",
                "validation_status": "failed",
                "repair_status": "not_started",
                "blockers": [],
                "failures": ["validation_failed"],
                "missing_outcomes": ["validation"],
                "expected_outcomes": {},
                "possible_next_actions": ["replan"],
                "continuation_hint": "replan",
                "bounded_history": [],
            },
            "expected_output": {},
            "evidence": None,
            "provenance": {},
            "metadata": {"quality_corpus": "negative", "policy_reject": "context_on_decision_context"},
        },
    },
    {
        "case_id": "pos_context_locate_intent",
        "expected": "accepted",
        "reason": "control positive — valid Context locate intent",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "capability_intent",
            "record_id": "pos-ctx-001",
            "system_contract": "execution.capability_intent",
            "provider_capability": "context",
            "input": {"objective": "Locate res.partner model", "reason": "control"},
            "expected_output": {
                "capability_id": "workspace.search",
                "args": {"query": "res.partner"},
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative_control_positive"},
        },
    },
)


# AT-7.3 — Conversation / Approval / Evaluation negative controls (NOT training data).
REASONING_SPARSE_NEGATIVE_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "neg_conv_forbidden_how",
        "expected": "rejected",
        "reason": "conversation clarify with forbidden HOW tokens",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "loop_decision",
            "record_id": "neg-conv-how-001",
            "system_contract": "intelligence_loop.decision",
            "provider_capability": "conversation",
            "expected_output": {
                "decision_kind": "clarify",
                "reason": "Ask user before running local_workspace and pytest",
                "next_goal": "Clarify path",
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_appr_forbidden_how",
        "expected": "rejected",
        "reason": "approval with forbidden HOW tokens",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "loop_decision",
            "record_id": "neg-appr-how-001",
            "system_contract": "intelligence_loop.decision",
            "provider_capability": "approval",
            "expected_output": {
                "decision_kind": "approve",
                "reason": "Approve after local_git sync and pytest green",
                "next_goal": "Proceed",
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_planner_as_conversation_policy",
        "expected": "accepted",
        "reason": (
            "planning_decision + conversation is schema-hydratable but mapping-forbidden "
            "(planning_decision providers = planner only); must not enter Conversation corpus"
        ),
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "planning_decision",
            "record_id": "neg-conv-plan-001",
            "system_contract": "intelligence.planning",
            "provider_capability": "conversation",
            "input": {"goal": "Ship feature"},
            "expected_output": {
                "decision_kind": "continue",
                "summary": "Plan next steps",
                "steps": [{"action": "workspace.read", "args": {"path": "a.py"}}],
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative", "policy_reject": "planner_as_conversation"},
        },
    },
    {
        "case_id": "neg_approval_as_conversation_policy",
        "expected": "accepted",
        "reason": (
            "approve under conversation is schema-valid; policy for Conversation corpus "
            "allows only clarify — must not enter Conversation corpus"
        ),
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "loop_decision",
            "record_id": "neg-conv-appr-001",
            "system_contract": "intelligence_loop.decision",
            "provider_capability": "conversation",
            "expected_output": {
                "decision_kind": "approve",
                "reason": "Looks good",
                "next_goal": "Continue",
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative", "policy_reject": "approval_as_conversation"},
        },
    },
    {
        "case_id": "neg_planner_as_approval_policy",
        "expected": "accepted",
        "reason": (
            "planning_decision + approval is schema-hydratable but mapping-forbidden; "
            "must not enter Approval corpus"
        ),
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "planning_decision",
            "record_id": "neg-appr-plan-001",
            "system_contract": "intelligence.planning",
            "provider_capability": "approval",
            "input": {"goal": "Ship"},
            "expected_output": {
                "decision_kind": "continue",
                "summary": "Plan",
                "steps": [],
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative", "policy_reject": "planner_as_approval"},
        },
    },
    {
        "case_id": "neg_conversation_as_approval_policy",
        "expected": "accepted",
        "reason": (
            "clarify under approval is schema-valid; Approval corpus allows only "
            "approve/reject/modify — must not enter Approval corpus"
        ),
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "loop_decision",
            "record_id": "neg-appr-conv-001",
            "system_contract": "intelligence_loop.decision",
            "provider_capability": "approval",
            "expected_output": {
                "decision_kind": "clarify",
                "reason": "Need more info",
                "next_goal": "Ask user",
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative", "policy_reject": "conversation_as_approval"},
        },
    },
    {
        "case_id": "neg_planner_as_evaluation_policy",
        "expected": "accepted",
        "reason": "planning_decision + evaluation mapping-forbidden; not Evaluation corpus",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "planning_decision",
            "record_id": "neg-eval-plan-001",
            "system_contract": "intelligence.planning",
            "provider_capability": "evaluation",
            "input": {"goal": "Ship"},
            "expected_output": {
                "decision_kind": "continue",
                "summary": "Plan",
                "steps": [],
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative", "policy_reject": "planner_as_evaluation"},
        },
    },
    {
        "case_id": "neg_approval_as_evaluation_policy",
        "expected": "accepted",
        "reason": "loop_decision not in evaluation mapping; must not be Evaluation corpus",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "loop_decision",
            "record_id": "neg-eval-appr-001",
            "system_contract": "intelligence_loop.decision",
            "provider_capability": "evaluation",
            "expected_output": {
                "decision_kind": "approve",
                "reason": "ok",
                "next_goal": "go",
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative", "policy_reject": "approval_as_evaluation"},
        },
    },
    {
        "case_id": "neg_eval_meta_judge_stuffed_decision_context",
        "expected": "accepted",
        "reason": (
            "decision_context + evaluation is schema-hydratable but mapping-rejected "
            "after AT-7.4 (evaluation only on evaluation_judgment); stuffing "
            "EvaluationRequest/Response fields into Continuity remains forbidden"
        ),
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "decision_context",
            "record_id": "neg-eval-meta-001",
            "system_contract": "execution.engineering_decision_context",
            "provider_capability": "evaluation",
            "input": {
                "objective": "Judge candidate plan",
                "objective_state": "incomplete",
                "cycle_index": 1,
                "execution_state": "idle",
                "observation_quality": "partial",
                "validation_status": "not_started",
                "repair_status": "not_started",
                "blockers": [],
                "failures": [],
                "missing_outcomes": ["judgment"],
                "expected_outcomes": {"verdict": "pass"},
                "possible_next_actions": ["evaluate"],
                "continuation_hint": "evaluate",
                "bounded_history": [],
            },
            "expected_output": {
                "candidate": {"plan": "fake"},
                "rubric": {"pass_if": "complete"},
                "verdict": "pass",
                "score": 0.9,
            },
            "evidence": None,
            "provenance": {},
            "metadata": {
                "quality_corpus": "negative",
                "policy_reject": "meta_judge_ambiguity",
            },
        },
    },
    {
        "case_id": "neg_eval_missing_candidate",
        "expected": "rejected",
        "reason": "evaluation_judgment without candidate",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "evaluation_judgment",
            "record_id": "neg-eval-cand-001",
            "system_contract": "capability.evaluation",
            "provider_capability": "evaluation",
            "input": {},
            "expected_output": {"verdict": "pass"},
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_eval_invalid_verdict",
        "expected": "rejected",
        "reason": "unknown evaluation verdict",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "evaluation_judgment",
            "record_id": "neg-eval-verdict-001",
            "system_contract": "capability.evaluation",
            "provider_capability": "evaluation",
            "input": {"candidate": {"answer": 1}},
            "expected_output": {"verdict": "maybe"},
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_eval_score_out_of_range",
        "expected": "rejected",
        "reason": "score outside [0,1]",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "evaluation_judgment",
            "record_id": "neg-eval-score-001",
            "system_contract": "capability.evaluation",
            "provider_capability": "evaluation",
            "input": {"candidate": {"answer": 1}},
            "expected_output": {"verdict": "pass", "score": 1.5},
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_eval_wrong_provider",
        "expected": "rejected",
        "reason": "evaluation_judgment requires provider_capability=evaluation",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "evaluation_judgment",
            "record_id": "neg-eval-provider-001",
            "system_contract": "capability.evaluation",
            "provider_capability": "planner",
            "input": {"candidate": {"answer": 1}},
            "expected_output": {"verdict": "pass"},
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_eval_forbidden_how",
        "expected": "rejected",
        "reason": "evaluation explanation with forbidden HOW token",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "evaluation_judgment",
            "record_id": "neg-eval-how-001",
            "system_contract": "capability.evaluation",
            "provider_capability": "evaluation",
            "input": {"candidate": {"answer": 1}},
            "expected_output": {
                "verdict": "fail",
                "explanation": "Reject because local_workspace was used",
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative"},
        },
    },
    {
        "case_id": "neg_conversation_as_evaluation_policy",
        "expected": "accepted",
        "reason": (
            "clarify loop_decision + evaluation is schema-hydratable but mapping-forbidden"
        ),
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "loop_decision",
            "record_id": "neg-eval-conv-001",
            "system_contract": "intelligence_loop.decision",
            "provider_capability": "evaluation",
            "expected_output": {
                "decision_kind": "clarify",
                "reason": "Ask question",
                "next_goal": "Clarify",
            },
            "provenance": {},
            "metadata": {
                "quality_corpus": "negative",
                "policy_reject": "conversation_as_evaluation",
            },
        },
    },
    {
        "case_id": "neg_execution_as_evaluation_policy",
        "expected": "accepted",
        "reason": (
            "execution_work_unit + evaluation is schema-hydratable but mapping-forbidden"
        ),
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "execution_work_unit",
            "record_id": "neg-eval-exec-001",
            "system_contract": "execution.work_unit",
            "provider_capability": "evaluation",
            "input": {"objective": "run", "inputs": {}, "constraints": {}},
            "expected_output": {
                "capability_id": "workspace.read",
                "work_id": "w1",
                "expected_outputs": {},
                "generated_code": "",
                "generated_artifacts": [],
                "validation": {},
            },
            "provenance": {},
            "metadata": {
                "quality_corpus": "negative",
                "policy_reject": "execution_as_evaluation",
            },
        },
    },
    {
        "case_id": "neg_observation_as_evaluation_policy",
        "expected": "accepted",
        "reason": (
            "observation + evaluation mapping-rejected after AT-7.4; not Evaluation corpus"
        ),
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "observation",
            "record_id": "neg-eval-obs-001",
            "system_contract": "execution.observation",
            "provider_capability": "evaluation",
            "evidence": {
                "kind": "generic",
                "status": "succeeded",
                "capability_id": "workspace.read",
                "summary": "ok",
                "details": {},
            },
            "provenance": {},
            "metadata": {
                "quality_corpus": "negative",
                "policy_reject": "observation_as_evaluation",
            },
        },
    },
    {
        "case_id": "pos_eval_judgment_control",
        "expected": "accepted",
        "reason": "control positive — valid evaluation_judgment",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "evaluation_judgment",
            "record_id": "pos-eval-001",
            "system_contract": "capability.evaluation",
            "provider_capability": "evaluation",
            "input": {
                "candidate": {"capability": "approval", "decision": {"status": "APPROVED"}},
                "expectation": {"capability": "approval", "decision": {"status": "APPROVED"}},
                "rubric": "Judge whether the approval candidate matches the expectation",
            },
            "expected_output": {
                "verdict": "pass",
                "score": 1.0,
                "explanation": "Candidate matches expectation",
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative_control_positive"},
        },
    },
    {
        "case_id": "pos_conv_clarify_control",
        "expected": "accepted",
        "reason": "control positive — valid conversation clarify",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "loop_decision",
            "record_id": "pos-conv-001",
            "system_contract": "intelligence_loop.decision",
            "provider_capability": "conversation",
            "expected_output": {
                "decision_kind": "clarify",
                "reason": "Ask which partner type requires VAT",
                "next_goal": "Clarify VAT scope",
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative_control_positive"},
        },
    },
    {
        "case_id": "pos_appr_approve_control",
        "expected": "accepted",
        "reason": "control positive — valid approval approve",
        "record": {
            "training_contract_version": "1.0.0",
            "record_type": "loop_decision",
            "record_id": "pos-appr-001",
            "system_contract": "intelligence_loop.decision",
            "provider_capability": "approval",
            "expected_output": {
                "decision_kind": "approve",
                "reason": "Discount within policy",
                "next_goal": "Confirm order",
            },
            "provenance": {},
            "metadata": {"quality_corpus": "negative_control_positive"},
        },
    },
)



def evaluate_negative_case(case: dict[str, Any]) -> dict[str, Any]:
    from aiodoo_training.system_training_contract.quality.gates import (
        provenance_ok_for_projected,
        scan_forbidden_how,
        scan_taxonomy,
    )
    from aiodoo_training.system_training_contract.records import (
        TrainingRecordError,
        validate_record_mapping,
    )

    record = case["record"]
    schema_ok = True
    schema_error = ""
    try:
        validate_record_mapping(record)
    except TrainingRecordError as exc:
        schema_ok = False
        schema_error = str(exc)

    how_issues = scan_forbidden_how(record)
    tax_issues = scan_taxonomy(record)
    prov_issues: list[str] = []
    if case.get("require_provenance") or (
        isinstance(record.get("metadata"), dict) and record["metadata"].get("projected")
    ):
        prov_issues = provenance_ok_for_projected(record)

    accepted = schema_ok and not how_issues and not tax_issues and not prov_issues
    expected = case["expected"]
    if expected == "accepted":
        matched = accepted
    elif expected == "rejected":
        matched = not accepted
    elif expected == "accepted_with_warn":
        matched = schema_ok  # semantic warn allowed
    else:
        matched = False

    return {
        "case_id": case["case_id"],
        "expected": expected,
        "matched": matched,
        "accepted": accepted,
        "schema_ok": schema_ok,
        "schema_error": schema_error,
        "how_issues": how_issues,
        "taxonomy_issues": tax_issues,
        "provenance_issues": prov_issues,
    }
