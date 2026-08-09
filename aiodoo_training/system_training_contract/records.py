"""Canonical Training record types (TR-2) — System WHAT surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

from aiodoo_training.system_training_contract.forbidden import (
    ForbiddenHowError,
    assert_no_forbidden_how,
)
from aiodoo_training.system_training_contract.taxonomy import (
    PREFERRED_ENGINEERING_CAPABILITY_IDS,
    TaxonomyPlane,
    classify_capability_id,
)
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

__all__ = [
    "RECORD_TYPES",
    "EVALUATION_VERDICTS",
    "TrainingRecordBase",
    "CapabilityIntentRecord",
    "WorkUnitRecord",
    "PlanningDecisionRecord",
    "ObservationRecord",
    "EngineeringFeedbackRecord",
    "EngineeringStateRecord",
    "DecisionContextRecord",
    "LoopDecisionRecord",
    "EvaluationJudgmentRecord",
    "validate_record_mapping",
    "TrainingRecordError",
]


class TrainingRecordError(ValueError):
    """Invalid canonical Training record."""


class FieldClass(StrEnum):
    MODEL_FACING_WHAT = "model_facing_what"
    SYSTEM_EVIDENCE = "system_evidence"
    HISTORICAL_METADATA = "historical_metadata"
    TRAINING_METADATA = "training_metadata"
    FORBIDDEN_HOW = "forbidden_how"


RECORD_TYPES: frozenset[str] = frozenset(
    {
        "capability_intent",
        "execution_work_unit",
        "planning_decision",
        "observation",
        "engineering_feedback",
        "engineering_state",
        "decision_context",
        "loop_decision",
        # AT-7.4 — mirrors aiodoo_contract EvaluationRequest/EvaluationResponse
        "evaluation_judgment",
    }
)

# Synchronized from aiodoo_contract.schemas.enums.EvaluationVerdict
EVALUATION_VERDICTS: frozenset[str] = frozenset({"pass", "fail", "inconclusive"})

# Synchronized from aiodoo.intelligence_loop.types.LoopDecisionKind, plus
# approval-plane reject/modify used by historical projection (not System enums).
LOOP_DECISION_KINDS: frozenset[str] = frozenset(
    {
        "continue",
        "retry",
        "clarify",
        "complete",
        "cancel",
        "escalate",
        "recover",
        "approve",
        "replan",
        "pause",
        # Approval / Training projection extras (provider-plane decisions)
        "reject",
        "modify",
    }
)

OBSERVATION_KINDS: frozenset[str] = frozenset(
    {
        "execution_result",
        "validation_result",
        "repair_result",
        "repository_change",
        "repository_status",
        "repository_comparison",
        "repository_history",
        "repository_branch",
        "repository_merge",
        "workspace_change",
        "artifact_result",
        "diagnostics_result",
        "search_result",
        "program_output",
        "environment_status",
        "capability_status",
        "execution_metrics",
        "planning_feedback",
        "diff_result",
        "generic",
    }
)

OBSERVATION_STATUSES: frozenset[str] = frozenset(
    {"succeeded", "failed", "partial", "skipped", "cancelled"}
)


@dataclass(frozen=True, slots=True)
class TrainingRecordBase:
    """Common envelope for all System Training Contract records."""

    record_type: str
    record_id: str
    training_contract_version: str = SYSTEM_TRAINING_CONTRACT_VERSION
    system_contract: str = ""  # e.g. execution.work_unit, intelligence_loop.decision
    provider_capability: str | None = None  # adapter pack id when applicable
    domain_specialization: str | None = None  # e.g. odoo | generic | None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def base_dict(self) -> dict[str, Any]:
        return {
            "training_contract_version": self.training_contract_version,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "system_contract": self.system_contract,
            "provider_capability": self.provider_capability,
            "domain_specialization": self.domain_specialization,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }

    def validate_base(self) -> None:
        if self.record_type not in RECORD_TYPES:
            raise TrainingRecordError(f"unknown record_type: {self.record_type!r}")
        if self.training_contract_version != SYSTEM_TRAINING_CONTRACT_VERSION:
            raise TrainingRecordError(
                f"unsupported training_contract_version: {self.training_contract_version!r}"
            )
        if not self.record_id.strip():
            raise TrainingRecordError("record_id is required")
        if self.provider_capability:
            plane = classify_capability_id(self.provider_capability)
            if plane is not TaxonomyPlane.PROVIDER:
                raise TrainingRecordError(
                    f"provider_capability must be a provider pack id, got "
                    f"{self.provider_capability!r} ({plane})"
                )


@dataclass(frozen=True, slots=True)
class CapabilityIntentRecord(TrainingRecordBase):
    """Model-facing Engineering capability intent (plan step / work action)."""

    capability_id: str = ""
    objective: str = ""
    args: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", "capability_intent")
        if not self.system_contract:
            object.__setattr__(self, "system_contract", "execution.capability_intent")

    def validate(self) -> None:
        self.validate_base()
        cid = self.capability_id.strip()
        if not cid:
            raise TrainingRecordError("capability_id is required")
        plane = classify_capability_id(cid)
        if plane is TaxonomyPlane.FORBIDDEN_HOW:
            raise TrainingRecordError(f"capability_id is forbidden HOW: {cid!r}")
        if plane is TaxonomyPlane.PROVIDER:
            raise TrainingRecordError(
                f"capability_id must be Engineering WHAT, not provider pack: {cid!r}"
            )
        # New Training records teach preferred FP2 domain.intent IDs only.
        # Transitional System aliases (shell, read, edit, …) are historical — not
        # emitted as canonical Training vocabulary.
        if cid not in PREFERRED_ENGINEERING_CAPABILITY_IDS:
            raise TrainingRecordError(
                f"capability_id must be preferred Engineering WHAT, got {cid!r}"
            )
        assert_no_forbidden_how(capability_id=cid, args=self.args)
        if not self.objective.strip():
            raise TrainingRecordError("objective is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.base_dict(),
            "input": {"objective": self.objective, "reason": self.reason},
            "expected_output": {
                "capability_id": self.capability_id,
                "args": dict(self.args),
            },
            "field_classes": {
                "capability_id": FieldClass.MODEL_FACING_WHAT.value,
                "args": FieldClass.MODEL_FACING_WHAT.value,
                "objective": FieldClass.MODEL_FACING_WHAT.value,
            },
        }


@dataclass(frozen=True, slots=True)
class WorkUnitRecord(TrainingRecordBase):
    """ExecutionWorkUnit Training surface — WHAT, never HOW."""

    capability_id: str = ""
    objective: str = ""
    work_id: str = ""
    inputs: Mapping[str, Any] = field(default_factory=dict)
    expected_outputs: Mapping[str, Any] = field(default_factory=dict)
    generated_code: str = ""
    generated_artifacts: tuple[str, ...] = ()
    validation: Mapping[str, Any] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", "execution_work_unit")
        if not self.system_contract:
            object.__setattr__(self, "system_contract", "execution.work_unit")
        if not self.work_id:
            object.__setattr__(self, "work_id", f"ewu-{uuid4().hex}")

    def validate(self) -> None:
        self.validate_base()
        cid = self.capability_id.strip()
        if cid not in PREFERRED_ENGINEERING_CAPABILITY_IDS:
            raise TrainingRecordError(
                f"WorkUnit capability_id must be preferred Engineering WHAT: {cid!r}"
            )
        assert_no_forbidden_how(capability_id=cid, args=self.inputs)
        assert_no_forbidden_how(args=self.expected_outputs)
        if not self.objective.strip():
            raise TrainingRecordError("objective is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.base_dict(),
            "input": {
                "objective": self.objective,
                "inputs": dict(self.inputs),
                "constraints": dict(self.constraints),
            },
            "expected_output": {
                "work_id": self.work_id,
                "capability_id": self.capability_id,
                "expected_outputs": dict(self.expected_outputs),
                "generated_code": self.generated_code,
                "generated_artifacts": list(self.generated_artifacts),
                "validation": dict(self.validation),
            },
            "field_classes": {
                "capability_id": FieldClass.MODEL_FACING_WHAT.value,
                "objective": FieldClass.MODEL_FACING_WHAT.value,
                "generated_code": FieldClass.MODEL_FACING_WHAT.value,
                "expected_outputs": FieldClass.MODEL_FACING_WHAT.value,
            },
        }


@dataclass(frozen=True, slots=True)
class PlanningDecisionRecord(TrainingRecordBase):
    """Planner / loop planning output: list of Engineering capability intents."""

    goal: str = ""
    summary: str = ""
    steps: tuple[Mapping[str, Any], ...] = ()
    decision_kind: str = "replan"  # typically leads to work; or complete with empty steps

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", "planning_decision")
        if not self.system_contract:
            object.__setattr__(self, "system_contract", "intelligence.planning")

    def validate(self) -> None:
        self.validate_base()
        if not self.goal.strip():
            raise TrainingRecordError("goal is required")
        kind = self.decision_kind.strip().lower()
        if kind not in LOOP_DECISION_KINDS:
            raise TrainingRecordError(f"unknown decision_kind: {self.decision_kind!r}")
        if kind == "complete" and self.steps:
            raise TrainingRecordError("COMPLETE planning must not include execution steps")
        for idx, step in enumerate(self.steps):
            if not isinstance(step, Mapping):
                raise TrainingRecordError(f"step {idx} must be a mapping")
            action = str(step.get("action") or step.get("capability_id") or "").strip()
            if not action:
                raise TrainingRecordError(f"step {idx} missing action/capability_id")
            if action not in PREFERRED_ENGINEERING_CAPABILITY_IDS:
                raise TrainingRecordError(
                    f"step {idx} action must be preferred Engineering capability, got {action!r}"
                )
            args = step.get("args") if isinstance(step.get("args"), Mapping) else {}
            assert_no_forbidden_how(capability_id=action, args=args)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.base_dict(),
            "input": {"goal": self.goal},
            "expected_output": {
                "decision_kind": self.decision_kind,
                "summary": self.summary,
                "steps": [dict(s) for s in self.steps],
            },
            "field_classes": {
                "steps": FieldClass.MODEL_FACING_WHAT.value,
                "decision_kind": FieldClass.MODEL_FACING_WHAT.value,
            },
        }


@dataclass(frozen=True, slots=True)
class ObservationRecord(TrainingRecordBase):
    """Public/model-facing observation envelope (backends stripped)."""

    kind: str = "generic"
    status: str = "succeeded"
    capability_id: str = ""
    summary: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", "observation")
        if not self.system_contract:
            object.__setattr__(self, "system_contract", "execution.observation")

    def validate(self) -> None:
        self.validate_base()
        if self.kind not in OBSERVATION_KINDS:
            raise TrainingRecordError(f"unknown observation kind: {self.kind!r}")
        if self.status not in OBSERVATION_STATUSES:
            raise TrainingRecordError(f"unknown observation status: {self.status!r}")
        if self.capability_id:
            if classify_capability_id(self.capability_id) is TaxonomyPlane.FORBIDDEN_HOW:
                raise TrainingRecordError("observation capability_id is forbidden HOW")
        assert_no_forbidden_how(args=self.evidence)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.base_dict(),
            "evidence": {
                "kind": self.kind,
                "status": self.status,
                "capability_id": self.capability_id,
                "summary": self.summary,
                "details": dict(self.evidence),
            },
            "field_classes": {
                "kind": FieldClass.SYSTEM_EVIDENCE.value,
                "status": FieldClass.SYSTEM_EVIDENCE.value,
                "details": FieldClass.SYSTEM_EVIDENCE.value,
            },
        }


@dataclass(frozen=True, slots=True)
class EngineeringFeedbackRecord(TrainingRecordBase):
    """EngineeringFeedback Training surface."""

    objective: str = ""
    objective_state: str = "unknown"
    execution_state: str = "unknown"
    observation_quality: str = "missing"
    continuation_options: tuple[str, ...] = ()
    recommended_continuation: str = ""
    observations: tuple[Mapping[str, Any], ...] = ()
    blockers: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    missing_outcomes: tuple[str, ...] = ()
    validation: Mapping[str, Any] = field(default_factory=dict)
    expected_outputs: Mapping[str, Any] = field(default_factory=dict)
    actual_outputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", "engineering_feedback")
        if not self.system_contract:
            object.__setattr__(self, "system_contract", "execution.engineering_feedback")

    def validate(self) -> None:
        self.validate_base()
        if not self.objective.strip():
            raise TrainingRecordError("objective is required")
        assert_no_forbidden_how(args=self.validation)
        assert_no_forbidden_how(args=self.expected_outputs)
        assert_no_forbidden_how(args=self.actual_outputs)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.base_dict(),
            "evidence": {
                "objective": self.objective,
                "objective_state": self.objective_state,
                "execution_state": self.execution_state,
                "observation_quality": self.observation_quality,
                "continuation_options": list(self.continuation_options),
                "recommended_continuation": self.recommended_continuation,
                "observations": [dict(o) for o in self.observations],
                "blockers": list(self.blockers),
                "failures": list(self.failures),
                "missing_outcomes": list(self.missing_outcomes),
                "validation": dict(self.validation),
                "expected_outputs": dict(self.expected_outputs),
                "actual_outputs": dict(self.actual_outputs),
            },
            "field_classes": {
                "objective_state": FieldClass.SYSTEM_EVIDENCE.value,
                "continuation_options": FieldClass.SYSTEM_EVIDENCE.value,
            },
        }


@dataclass(frozen=True, slots=True)
class EngineeringStateRecord(TrainingRecordBase):
    """Current-cycle EngineeringState projection (not Memory history dump)."""

    objective: str = ""
    objective_state: str = "unknown"
    session_state: str = ""
    completion_state: str = ""
    cycle_index: int = 0
    current_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", "engineering_state")
        if not self.system_contract:
            object.__setattr__(self, "system_contract", "execution.engineering_state")

    def validate(self) -> None:
        self.validate_base()
        if self.cycle_index < 0:
            raise TrainingRecordError("cycle_index must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.base_dict(),
            "evidence": {
                "objective": self.objective,
                "objective_state": self.objective_state,
                "session_state": self.session_state,
                "completion_state": self.completion_state,
                "cycle_index": self.cycle_index,
                "current_fields": dict(self.current_fields),
            },
            "field_classes": {
                "current_fields": FieldClass.SYSTEM_EVIDENCE.value,
            },
        }


@dataclass(frozen=True, slots=True)
class DecisionContextRecord(TrainingRecordBase):
    """EngineeringDecisionContext — bounded model-facing bridge."""

    objective: str = ""
    objective_state: str = "unknown"
    cycle_index: int = 0
    execution_state: str = ""
    observation_quality: str = ""
    validation_status: str = ""
    repair_status: str = ""
    expected_outcomes: Mapping[str, Any] = field(default_factory=dict)
    missing_outcomes: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    possible_next_actions: tuple[str, ...] = ()
    continuation_hint: str = ""
    bounded_history: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", "decision_context")
        if not self.system_contract:
            object.__setattr__(
                self, "system_contract", "execution.engineering_decision_context"
            )

    def validate(self) -> None:
        self.validate_base()
        if not self.objective.strip():
            raise TrainingRecordError("objective is required")
        assert_no_forbidden_how(args=self.expected_outcomes)
        for hist in self.bounded_history:
            if not isinstance(hist, Mapping):
                raise TrainingRecordError("bounded_history entries must be mappings")
            assert_no_forbidden_how(args=hist)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.base_dict(),
            "input": {
                "objective": self.objective,
                "objective_state": self.objective_state,
                "cycle_index": self.cycle_index,
                "execution_state": self.execution_state,
                "observation_quality": self.observation_quality,
                "validation_status": self.validation_status,
                "repair_status": self.repair_status,
                "expected_outcomes": dict(self.expected_outcomes),
                "missing_outcomes": list(self.missing_outcomes),
                "blockers": list(self.blockers),
                "failures": list(self.failures),
                "possible_next_actions": list(self.possible_next_actions),
                "continuation_hint": self.continuation_hint,
                "bounded_history": [dict(h) for h in self.bounded_history],
            },
            "field_classes": {
                "input": FieldClass.MODEL_FACING_WHAT.value,
            },
        }


@dataclass(frozen=True, slots=True)
class LoopDecisionRecord(TrainingRecordBase):
    """Intelligence Loop decision: REPLAN / COMPLETE / ESCALATE / …"""

    decision_kind: str = ""
    reason: str = ""
    next_goal: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", "loop_decision")
        if not self.system_contract:
            object.__setattr__(self, "system_contract", "intelligence_loop.decision")

    def validate(self) -> None:
        self.validate_base()
        kind = self.decision_kind.strip().lower()
        if kind not in LOOP_DECISION_KINDS:
            raise TrainingRecordError(f"unknown loop decision_kind: {self.decision_kind!r}")
        if kind in {"replan", "complete", "escalate"} and not self.reason.strip():
            raise TrainingRecordError("reason is required for REPLAN/COMPLETE/ESCALATE")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.base_dict(),
            "expected_output": {
                "decision_kind": self.decision_kind.strip().lower(),
                "reason": self.reason,
                "next_goal": self.next_goal,
            },
            "field_classes": {
                "decision_kind": FieldClass.MODEL_FACING_WHAT.value,
            },
        }


@dataclass(frozen=True, slots=True)
class EvaluationJudgmentRecord(TrainingRecordBase):
    """Provider-plane Evaluation judgment (AT-7.4).

    Mirrors ``aiodoo_contract.schemas.evaluation.EvaluationRequest`` →
    ``EvaluationResponse``. Not an Engineering WHAT action. Not Runtime
    validation infrastructure. Candidate payloads are generic dicts so any
    capability output (plan, code, approval, conversation, tool result, …)
    may be judged without separate per-capability record families.
    """

    candidate: Mapping[str, Any] = field(default_factory=dict)
    expectation: Mapping[str, Any] | None = None
    rubric: str | None = None
    verdict: str = ""
    score: float | None = None
    explanation: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", "evaluation_judgment")
        if not self.system_contract:
            object.__setattr__(self, "system_contract", "capability.evaluation")

    def validate(self) -> None:
        self.validate_base()
        if self.provider_capability != "evaluation":
            raise TrainingRecordError(
                "evaluation_judgment requires provider_capability='evaluation'"
            )
        if not isinstance(self.candidate, Mapping) or not dict(self.candidate):
            raise TrainingRecordError("candidate is required and must be a non-empty mapping")
        if self.expectation is not None and not isinstance(self.expectation, Mapping):
            raise TrainingRecordError("expectation must be a mapping when provided")
        if self.rubric is not None and not isinstance(self.rubric, str):
            raise TrainingRecordError("rubric must be a string when provided")
        verdict = self.verdict.strip().lower()
        if verdict not in EVALUATION_VERDICTS:
            raise TrainingRecordError(f"unknown evaluation verdict: {self.verdict!r}")
        if self.score is not None:
            try:
                score_f = float(self.score)
            except (TypeError, ValueError) as exc:
                raise TrainingRecordError(f"invalid evaluation score: {self.score!r}") from exc
            if score_f < 0.0 or score_f > 1.0:
                raise TrainingRecordError(
                    f"evaluation score must be in [0.0, 1.0], got {score_f}"
                )
        assert_no_forbidden_how(args=dict(self.candidate))
        if self.expectation is not None:
            assert_no_forbidden_how(args=dict(self.expectation))
        text_fields: dict[str, str] = {
            "candidate": str(dict(self.candidate)),
        }
        if self.expectation is not None:
            text_fields["expectation"] = str(dict(self.expectation))
        if self.rubric:
            text_fields["rubric"] = self.rubric
        if self.explanation:
            text_fields["explanation"] = self.explanation
        assert_no_forbidden_how(text_fields=text_fields)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        inp: dict[str, Any] = {"candidate": dict(self.candidate)}
        if self.expectation is not None:
            inp["expectation"] = dict(self.expectation)
        if self.rubric is not None:
            inp["rubric"] = self.rubric
        out: dict[str, Any] = {
            "verdict": self.verdict.strip().lower(),
        }
        if self.score is not None:
            out["score"] = float(self.score)
        if self.explanation is not None:
            out["explanation"] = self.explanation
        return {
            **self.base_dict(),
            "input": inp,
            "expected_output": out,
            "field_classes": {
                "input": FieldClass.MODEL_FACING_WHAT.value,
                "expected_output": FieldClass.MODEL_FACING_WHAT.value,
            },
        }


_RECORD_BUILDERS = {
    "capability_intent": CapabilityIntentRecord,
    "execution_work_unit": WorkUnitRecord,
    "planning_decision": PlanningDecisionRecord,
    "observation": ObservationRecord,
    "engineering_feedback": EngineeringFeedbackRecord,
    "engineering_state": EngineeringStateRecord,
    "decision_context": DecisionContextRecord,
    "loop_decision": LoopDecisionRecord,
    "evaluation_judgment": EvaluationJudgmentRecord,
}


def validate_record_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a serialized canonical Training record mapping."""
    if not isinstance(data, Mapping):
        raise TrainingRecordError("record must be a mapping")
    rtype = str(data.get("record_type") or "").strip()
    if rtype not in _RECORD_BUILDERS:
        raise TrainingRecordError(f"unknown or missing record_type: {rtype!r}")
    # Lightweight structural re-hydrate for validation only.
    base_kwargs = {
        "record_type": rtype,
        "record_id": str(data.get("record_id") or ""),
        "training_contract_version": str(
            data.get("training_contract_version") or SYSTEM_TRAINING_CONTRACT_VERSION
        ),
        "system_contract": str(data.get("system_contract") or ""),
        "provider_capability": data.get("provider_capability"),
        "domain_specialization": data.get("domain_specialization"),
        "provenance": dict(data.get("provenance") or {}),
        "metadata": dict(data.get("metadata") or {}),
    }
    expected = data.get("expected_output") if isinstance(data.get("expected_output"), Mapping) else {}
    evidence = data.get("evidence") if isinstance(data.get("evidence"), Mapping) else {}
    inp = data.get("input") if isinstance(data.get("input"), Mapping) else {}

    try:
        if rtype == "capability_intent":
            rec = CapabilityIntentRecord(
                **base_kwargs,
                capability_id=str(expected.get("capability_id") or ""),
                objective=str(inp.get("objective") or ""),
                args=dict(expected.get("args") or {}),
                reason=str(inp.get("reason") or ""),
            )
        elif rtype == "execution_work_unit":
            rec = WorkUnitRecord(
                **base_kwargs,
                capability_id=str(expected.get("capability_id") or ""),
                objective=str(inp.get("objective") or ""),
                work_id=str(expected.get("work_id") or ""),
                inputs=dict(inp.get("inputs") or {}),
                expected_outputs=dict(expected.get("expected_outputs") or {}),
                generated_code=str(expected.get("generated_code") or ""),
                generated_artifacts=tuple(expected.get("generated_artifacts") or ()),
                validation=dict(expected.get("validation") or {}),
                constraints=dict(inp.get("constraints") or {}),
            )
        elif rtype == "planning_decision":
            rec = PlanningDecisionRecord(
                **base_kwargs,
                goal=str(inp.get("goal") or ""),
                summary=str(expected.get("summary") or ""),
                steps=tuple(expected.get("steps") or ()),
                decision_kind=str(expected.get("decision_kind") or "replan"),
            )
        elif rtype == "observation":
            rec = ObservationRecord(
                **base_kwargs,
                kind=str(evidence.get("kind") or "generic"),
                status=str(evidence.get("status") or "succeeded"),
                capability_id=str(evidence.get("capability_id") or ""),
                summary=str(evidence.get("summary") or ""),
                evidence=dict(evidence.get("details") or {}),
            )
        elif rtype == "engineering_feedback":
            rec = EngineeringFeedbackRecord(
                **base_kwargs,
                objective=str(evidence.get("objective") or ""),
                objective_state=str(evidence.get("objective_state") or "unknown"),
                execution_state=str(evidence.get("execution_state") or "unknown"),
                observation_quality=str(evidence.get("observation_quality") or "missing"),
                continuation_options=tuple(evidence.get("continuation_options") or ()),
                recommended_continuation=str(evidence.get("recommended_continuation") or ""),
                observations=tuple(evidence.get("observations") or ()),
                blockers=tuple(evidence.get("blockers") or ()),
                failures=tuple(evidence.get("failures") or ()),
                missing_outcomes=tuple(evidence.get("missing_outcomes") or ()),
                validation=dict(evidence.get("validation") or {}),
                expected_outputs=dict(evidence.get("expected_outputs") or {}),
                actual_outputs=dict(evidence.get("actual_outputs") or {}),
            )
        elif rtype == "engineering_state":
            rec = EngineeringStateRecord(
                **base_kwargs,
                objective=str(evidence.get("objective") or ""),
                objective_state=str(evidence.get("objective_state") or "unknown"),
                session_state=str(evidence.get("session_state") or ""),
                completion_state=str(evidence.get("completion_state") or ""),
                cycle_index=int(evidence.get("cycle_index") or 0),
                current_fields=dict(evidence.get("current_fields") or {}),
            )
        elif rtype == "decision_context":
            rec = DecisionContextRecord(
                **base_kwargs,
                objective=str(inp.get("objective") or ""),
                objective_state=str(inp.get("objective_state") or "unknown"),
                cycle_index=int(inp.get("cycle_index") or 0),
                execution_state=str(inp.get("execution_state") or ""),
                observation_quality=str(inp.get("observation_quality") or ""),
                validation_status=str(inp.get("validation_status") or ""),
                repair_status=str(inp.get("repair_status") or ""),
                expected_outcomes=dict(inp.get("expected_outcomes") or {}),
                missing_outcomes=tuple(inp.get("missing_outcomes") or ()),
                blockers=tuple(inp.get("blockers") or ()),
                failures=tuple(inp.get("failures") or ()),
                possible_next_actions=tuple(inp.get("possible_next_actions") or ()),
                continuation_hint=str(inp.get("continuation_hint") or ""),
                bounded_history=tuple(inp.get("bounded_history") or ()),
            )
        elif rtype == "loop_decision":
            rec = LoopDecisionRecord(
                **base_kwargs,
                decision_kind=str(expected.get("decision_kind") or ""),
                reason=str(expected.get("reason") or ""),
                next_goal=str(expected.get("next_goal") or ""),
            )
        else:  # evaluation_judgment
            score_raw = expected.get("score")
            score_val: float | None
            if score_raw is None or score_raw == "":
                score_val = None
            else:
                score_val = float(score_raw)
            expectation_raw = inp.get("expectation")
            expectation_val: Mapping[str, Any] | None
            if expectation_raw is None:
                expectation_val = None
            elif isinstance(expectation_raw, Mapping):
                expectation_val = expectation_raw
            else:
                raise TrainingRecordError("expectation must be a mapping when provided")
            rubric_raw = inp.get("rubric")
            rubric_val = None if rubric_raw is None else str(rubric_raw)
            explanation_raw = expected.get("explanation")
            explanation_val = None if explanation_raw is None else str(explanation_raw)
            candidate_raw = inp.get("candidate")
            if not isinstance(candidate_raw, Mapping):
                candidate_raw = {}
            rec = EvaluationJudgmentRecord(
                **base_kwargs,
                candidate=candidate_raw,
                expectation=expectation_val,
                rubric=rubric_val,
                verdict=str(expected.get("verdict") or ""),
                score=score_val,
                explanation=explanation_val,
            )
        return rec.to_dict()
    except (ForbiddenHowError, TrainingRecordError, TypeError, ValueError) as exc:
        raise TrainingRecordError(str(exc)) from exc
