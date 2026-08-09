"""AT-7.4 — Evaluation provider semantic + FP2 mapping decision.

System EvaluationRequest/Response is already defined in aiodoo-contract.
Training Contract adds ``evaluation_judgment`` as the sole FP2 family for
``provider_capability=evaluation``. No corpus generation in this module.
"""

from __future__ import annotations

from typing import Any

EVALUATION_CONTRACT_DECISION: str = "EVALUATION_MAPPING_READY"

EVALUATION_SEMANTIC_DEFINITION: dict[str, Any] = {
    "status": EVALUATION_CONTRACT_DECISION,
    "at73_status": "EVALUATION_SEMANTICS_UNRESOLVED",
    "system_capability_resolved": True,
    "fp2_representation_resolved": True,
    "fp2_generation_authorized": True,
    "fp2_corpus_version": "fp2-evaluation-controlled-1.0.0",
    "fp2_corpus_verdict": "EVALUATION_CORPUS_READY",
    "definition": (
        "Evaluation is the Reasoning provider capability that judges a generic "
        "candidate (typically another capability's request/response dump) against "
        "optional expectation/rubric and returns verdict, optional score [0,1], "
        "and optional explanation. Training teaches this structured judgment "
        "behavior; Runtime/Agents execute EvaluationRequest handling."
    ),
    "system_contract": "capability.evaluation",
    "record_type": "evaluation_judgment",
    "evidence": [
        "aiodoo_contract.schemas.evaluation.EvaluationRequest "
        "(candidate required; expectation?, rubric?)",
        "aiodoo_contract.schemas.evaluation.EvaluationResponse "
        "(verdict: pass|fail|inconclusive; score? [0,1]; explanation?)",
        "aiodoo_contract.schemas.enums.EvaluationVerdict",
        "aiodoo_contract.foundations.REASONING_CAPABILITIES includes evaluation",
        "aiodoo_contract.prompts.capability._extract_evaluation",
        "docs/capability_model.md: evaluation = judgment SFT; not BenchmarkCatalog",
        "legacy evaluation_dataset.jsonl matches meta-judge request/response shape",
    ],
    "answers": {
        "learns": (
            "Emit EvaluationResponse-shaped judgments given EvaluationRequest-shaped inputs"
        ),
        "inputs": "candidate (+ optional expectation, rubric)",
        "outputs": "verdict, optional score [0,1], optional explanation",
        "judges_other_capability": True,
        "produces_rubric_verdict_score": True,
        "evaluates_candidate_actions": (
            "Yes — when candidate payload is an action/capability output dump"
        ),
        "evaluates_generated_code": "Yes — when candidate is coding/repair output",
        "evaluates_plans": "Yes — when candidate is planner output",
        "evaluates_tool_results": "Yes — when candidate encodes tool/observation output",
        "evaluates_conversations": "Yes — when candidate is conversation output",
        "candidate_representation": (
            "Generic dict payload on evaluation_judgment — NOT separate families "
            "per judged capability"
        ),
        "fp2_families_authorized": ["evaluation_judgment"],
        "fp2_families_rejected": [
            "capability_intent",
            "execution_work_unit",
            "planning_decision",
            "observation",
            "engineering_feedback",
            "engineering_state",
            "decision_context",
            "loop_decision",
        ],
        "engineering_what": "none — Evaluation is provider-plane judgment",
    },
    "mapping_decision": {
        "allowed": ["evaluation_judgment"],
        "removed_from": [
            "observation",
            "engineering_feedback",
            "decision_context",
        ],
        "reason": (
            "Existing Continuity/observation/feedback families do not encode "
            "EvaluationRequest/Response. Forcing judgment fields into them "
            "would fabricate a Training Contract surface."
        ),
    },
    "legacy_projection": "NOT PERFORMED — recommend future Evaluation Legacy → FP2 Projection",
    "next_phase": "AT-7.6 Controlled Evaluation Skill Adapter Path Smoke",
}
