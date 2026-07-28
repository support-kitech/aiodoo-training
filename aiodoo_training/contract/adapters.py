"""Project a protocol JSONL record onto its canonical `aiodoo_contract` shape.

Mirrors the projection performed on the producer side by
aiodoo-datasets' ``generators/common/contract/adapters.py`` against the
same record shape and the same target schemas — see the package docstring
for why this is re-implemented here rather than imported. Field-mapping
decisions (which record keys feed which schema field, how dataset-specific
enums map onto contract enums) are kept identical to that module so a
record projects the same way regardless of which repository is reading it.

Every ``project_*`` function raises :class:`ContractAdapterError` (never a
bare ``KeyError``/``TypeError``) when a record cannot be projected, so
formatters can treat "this record does not have enough structure to build
a contract-shaped training example" as a distinct, expected outcome.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from aiodoo_contract.schemas.approval import ApprovalRequest, ApprovalResponse
from aiodoo_contract.schemas.base import CapabilityRequest, CapabilityResponse
from aiodoo_contract.schemas.coding import CodingRequest, CodingResponse, FileEdit
from aiodoo_contract.schemas.conversation import (
    ConversationRequest,
    ConversationResponse,
    ConversationTurn,
)
from aiodoo_contract.schemas.enums import (
    ApprovalStatus,
    ConversationRole,
    EvaluationVerdict,
    ExecutionStatus,
)
from aiodoo_contract.schemas.evaluation import EvaluationRequest, EvaluationResponse
from aiodoo_contract.schemas.execution import ExecutionRequest, ExecutionResponse
from aiodoo_contract.schemas.planner import PlannerRequest, PlannerResponse, PlanStep
from aiodoo_contract.schemas.repair import RepairFix, RepairRequest, RepairResponse

__all__ = [
    "SUPPORTED_CAPABILITIES",
    "ContractAdapterError",
    "ContractProjection",
    "project_approval",
    "project_coding",
    "project_conversation",
    "project_evaluation",
    "project_execution",
    "project_planner",
    "project_record",
    "project_repair",
]


class ContractAdapterError(ValueError):
    """A record does not carry enough structure to project onto the contract.

    This is an expected, recoverable outcome — callers should catch this
    specifically and fall back rather than letting it propagate as a bare
    ``KeyError``/``TypeError``.
    """


@dataclass(frozen=True, slots=True)
class ContractProjection:
    """A record's canonical ``aiodoo_contract`` request/response projection."""

    capability: str
    request: CapabilityRequest
    response: CapabilityResponse


def _require_dict(value: object, message: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ContractAdapterError(message)
    return dict(value)


def _require_str(value: object, message: str) -> str:
    if value is None:
        raise ContractAdapterError(message)
    text = str(value).strip()
    if not text:
        raise ContractAdapterError(message)
    return text


# ---------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------


def project_planner(record: Mapping[str, Any]) -> ContractProjection:
    """Project a planner dataset record onto `PlannerRequest`/`PlannerResponse`."""
    output = _require_dict(record.get("output"), "planner record is missing 'output'")
    goal = _require_str(
        output.get("goal") or record.get("instruction"),
        "planner record has no usable goal (output.goal / instruction)",
    )
    tasks = output.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ContractAdapterError("planner record has no tasks to project into plan steps")

    steps: list[PlanStep] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            continue
        description = task.get("title") or task.get("description")
        if not description:
            continue
        steps.append(
            PlanStep(
                index=index,
                description=str(description),
                capability=None,
                inputs={
                    "task_id": str(task.get("id", "")),
                    "priority": str(task.get("priority", "")),
                },
            )
        )
    if not steps:
        raise ContractAdapterError("planner record's tasks had no usable title/description")

    context = record.get("input")
    request = PlannerRequest(goal=goal, context=str(context) if context else None)
    response = PlannerResponse(request_id=request.request_id, steps=steps)
    return ContractProjection("planner", request, response)


# ---------------------------------------------------------------------
# Coding
# ---------------------------------------------------------------------


def project_coding(record: Mapping[str, Any]) -> ContractProjection:
    """Project a coding dataset record onto `CodingRequest`/`CodingResponse`."""
    output = _require_dict(record.get("output"), "coding record is missing 'output'")
    instruction = _require_str(record.get("instruction"), "coding record is missing 'instruction'")

    artifacts = output.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ContractAdapterError("coding record has no artifacts to project into edits")

    edits: list[FileEdit] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        path = artifact.get("path")
        if not path:
            continue
        content = artifact.get("diff") or artifact.get("content") or ""
        edits.append(
            FileEdit(
                path=str(path),
                content=str(content),
                change_summary=(str(artifact["reason"]) if artifact.get("reason") else None),
            )
        )
    if not edits:
        raise ContractAdapterError("coding record's artifacts had no usable 'path'")

    request = CodingRequest(instruction=instruction)
    response = CodingResponse(
        request_id=request.request_id,
        edits=edits,
        rationale=(str(output["goal"]) if output.get("goal") else None),
    )
    return ContractProjection("coding", request, response)


# ---------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------

_SEVERITY_CONFIDENCE = {
    "critical": 0.6,
    "high": 0.65,
    "medium": 0.75,
    "low": 0.85,
    "info": 0.9,
}


def _apply_search_replace(content: str, operations: object) -> str:
    """Apply a repair task's ``search``/``replace`` operations to ``content``."""
    result = content
    if not isinstance(operations, list):
        return result
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        search = operation.get("search")
        replace = operation.get("replace")
        if not isinstance(search, str) or not isinstance(replace, str):
            continue
        result = result.replace(search, replace)
    return result


def project_repair(record: Mapping[str, Any]) -> ContractProjection:
    """Project a repair dataset record onto `RepairRequest`/`RepairResponse`."""
    output = _require_dict(record.get("output"), "repair record is missing 'output'")
    tasks = output.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ContractAdapterError("repair record has no tasks to project a fix from")

    descriptions: list[str] = []
    diagnostics: list[str] = []
    severities: list[str] = []
    edits: list[FileEdit] = []

    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        raw_problem = task.get("problem")
        problem: Mapping[str, Any] = raw_problem if isinstance(raw_problem, Mapping) else {}
        raw_root_cause = task.get("root_cause")
        root_cause: Mapping[str, Any] = (
            raw_root_cause if isinstance(raw_root_cause, Mapping) else {}
        )
        if problem.get("description"):
            descriptions.append(str(problem["description"]))
        if problem.get("severity"):
            severities.append(str(problem["severity"]).lower())
        if root_cause.get("analysis"):
            diagnostics.append(str(root_cause["analysis"]))

        artifacts = task.get("artifacts")
        raw_expected_outcome = task.get("expected_outcome")
        operations = (
            raw_expected_outcome.get("operations")
            if isinstance(raw_expected_outcome, Mapping)
            else None
        )
        if isinstance(artifacts, list) and artifacts:
            primary = artifacts[0]
            if isinstance(primary, Mapping) and primary.get("path"):
                original = str(primary.get("content", ""))
                new_content = _apply_search_replace(original, operations)
                op_count = len(operations) if isinstance(operations, list) else 0
                summary = (f"{op_count} operation(s) for: {problem.get('description', '')}").strip()
                edits.append(
                    FileEdit(
                        path=str(primary["path"]),
                        content=new_content,
                        change_summary=summary[:500] or None,
                    )
                )

    if not edits:
        raise ContractAdapterError(
            "repair record's tasks had no artifact with a 'path' to project a fix onto"
        )

    failure_description = " ".join(descriptions) or record.get("instruction")
    failure_description = _require_str(
        failure_description, "repair record has no failure description to project"
    )
    diagnostic_context = "; ".join(diagnostics) or None

    confidence = (
        sum(_SEVERITY_CONFIDENCE.get(s, 0.7) for s in severities) / len(severities)
        if severities
        else 0.7
    )

    fix = RepairFix(description=failure_description, edits=edits, confidence=confidence)
    request = RepairRequest(
        failure_description=failure_description, diagnostic_context=diagnostic_context
    )
    response = RepairResponse(request_id=request.request_id, fix=fix)
    return ContractProjection("repair", request, response)


# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------


def project_execution(record: Mapping[str, Any]) -> ContractProjection:
    """Project an execution dataset record onto `ExecutionRequest`/`ExecutionResponse`."""
    output = _require_dict(record.get("output"), "execution record is missing 'output'")
    instruction = _require_str(
        record.get("instruction"), "execution record is missing 'instruction'"
    )

    module = output.get("module")
    request = ExecutionRequest(
        command=instruction,
        arguments=[],
        working_directory=str(module) if module else None,
    )

    steps = output.get("steps")
    has_steps = isinstance(steps, list) and len(steps) > 0
    status = ExecutionStatus.SUCCEEDED if has_steps else ExecutionStatus.PENDING
    summary = output.get("summary") or ""
    response = ExecutionResponse(
        request_id=request.request_id,
        status=status,
        exit_code=0 if has_steps else None,
        stdout=str(summary),
        stderr="",
    )
    return ContractProjection("execution", request, response)


# ---------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------

# "reviewer" has no aiodoo_contract.ConversationRole equivalent; folded into
# ASSISTANT (a reviewer turn is agent-side, not user-side) — matches the
# same documented decision in aiodoo-datasets' adapter.
_ROLE_MAP = {
    "system": ConversationRole.SYSTEM,
    "user": ConversationRole.USER,
    "assistant": ConversationRole.ASSISTANT,
    "tool": ConversationRole.TOOL,
    "reviewer": ConversationRole.ASSISTANT,
}


def project_conversation(record: Mapping[str, Any]) -> ContractProjection:
    """Project a conversation dataset record onto Conversation Request/Response."""
    output = _require_dict(record.get("output"), "conversation record is missing 'output'")
    turns_data = output.get("turns")
    if not isinstance(turns_data, list) or not turns_data:
        raise ContractAdapterError("conversation record has no turns")

    messages: list[Mapping[str, Any]] = []
    for turn in turns_data:
        if not isinstance(turn, Mapping):
            continue
        for message in turn.get("messages", []):
            if isinstance(message, Mapping) and message.get("role") and message.get("content"):
                messages.append(message)

    if len(messages) < 2:
        raise ContractAdapterError(
            "conversation record has fewer than 2 messages; cannot split into request turns + reply"
        )

    request_turns: list[ConversationTurn] = []
    for message in messages[:-1]:
        role = _ROLE_MAP.get(str(message["role"]).lower())
        if role is None:
            continue
        request_turns.append(ConversationTurn(role=role, content=str(message["content"])))
    if not request_turns:
        raise ContractAdapterError(
            "conversation record's messages had no role mappable to ConversationRole"
        )

    last = messages[-1]
    reply_role = _ROLE_MAP.get(str(last["role"]).lower(), ConversationRole.ASSISTANT)
    reply = ConversationTurn(role=reply_role, content=str(last["content"]))

    request = ConversationRequest(turns=request_turns)
    response = ConversationResponse(request_id=request.request_id, reply=reply)
    return ContractProjection("conversation", request, response)


# ---------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------

# The dataset's own decision enum has a CHANGES_REQUESTED outcome with no
# equivalent in the contract's ApprovalStatus (pending/approved/rejected).
# Mapped to PENDING (the closest non-terminal state) — a documented,
# intentional divergence, not a claim the two are equivalent.
_DECISION_STATUS_MAP = {
    "APPROVED": ApprovalStatus.APPROVED,
    "REJECTED": ApprovalStatus.REJECTED,
    "CHANGES_REQUESTED": ApprovalStatus.PENDING,
}


def project_approval(record: Mapping[str, Any]) -> ContractProjection:
    """Project an approval dataset record onto `ApprovalRequest`/`ApprovalResponse`."""
    decision = _require_dict(record.get("decision"), "approval record is missing 'decision'")
    status_raw = str(decision.get("status", "")).upper()
    status = _DECISION_STATUS_MAP.get(status_raw)
    if status is None:
        raise ContractAdapterError(
            f"approval record has an unmappable decision status: {status_raw!r}"
        )

    raw_metadata = record.get("metadata")
    metadata: Mapping[str, Any] = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    module = metadata.get("source_module") or metadata.get("module") or "unknown module"
    evidence = record.get("evidence")
    review_id = record.get("review_id") or ""

    request = ApprovalRequest(
        subject=f"Approval review for {module}",
        payload={
            "review_id": str(review_id),
            "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
        },
    )
    reason = decision.get("reasoning")
    response = ApprovalResponse(
        request_id=request.request_id,
        status=status,
        reason=str(reason) if reason else None,
    )
    return ContractProjection("approval", request, response)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

_VERDICT_MAP = {
    "pass": EvaluationVerdict.PASS,
    "fail": EvaluationVerdict.FAIL,
    "inconclusive": EvaluationVerdict.INCONCLUSIVE,
}


def project_evaluation(record: Mapping[str, Any]) -> ContractProjection:
    """Project an Evaluation SFT judgment onto `EvaluationRequest`/`EvaluationResponse`.

    Field mapping matches aiodoo-datasets'
    ``generators.common.contract.adapters.project_evaluation`` so a judgment
    record projects identically on the producer and consumer sides.
    """
    candidate_raw = record.get("candidate")
    if not isinstance(candidate_raw, Mapping) or not candidate_raw:
        raise ContractAdapterError("evaluation record is missing 'candidate'")
    candidate = dict(candidate_raw)

    expectation_raw = record.get("expectation")
    expectation: dict[str, object] | None
    if expectation_raw is None:
        expectation = None
    elif isinstance(expectation_raw, Mapping):
        expectation = dict(expectation_raw)
    else:
        raise ContractAdapterError("evaluation record has invalid 'expectation'")

    rubric_raw = record.get("rubric")
    rubric = str(rubric_raw) if isinstance(rubric_raw, str) and rubric_raw.strip() else None

    verdict_raw = str(record.get("verdict", "")).strip().lower()
    verdict = _VERDICT_MAP.get(verdict_raw)
    if verdict is None:
        raise ContractAdapterError(f"evaluation record has an unmappable verdict: {verdict_raw!r}")

    score_raw = record.get("score")
    score: float | None
    if score_raw is None:
        score = None
    else:
        try:
            score = float(score_raw)
        except (TypeError, ValueError) as exc:
            raise ContractAdapterError("evaluation record has invalid 'score'") from exc
        if score < 0.0 or score > 1.0:
            raise ContractAdapterError("evaluation record score must be in [0, 1]")

    explanation_raw = record.get("explanation")
    explanation = (
        str(explanation_raw)
        if isinstance(explanation_raw, str) and explanation_raw.strip()
        else None
    )

    request = EvaluationRequest(candidate=candidate, expectation=expectation, rubric=rubric)
    response = EvaluationResponse(
        request_id=request.request_id,
        verdict=verdict,
        score=score,
        explanation=explanation,
    )
    return ContractProjection("evaluation", request, response)


# ---------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------

_PROJECTORS: dict[str, Callable[[Mapping[str, Any]], ContractProjection]] = {
    "planner": project_planner,
    "coding": project_coding,
    "repair": project_repair,
    "execution": project_execution,
    "conversation": project_conversation,
    "approval": project_approval,
    "evaluation": project_evaluation,
}

#: Dataset types with a canonical `aiodoo_contract` projection. ``context``
#: and ``mixed`` have no projection here — they are not capability
#: request/response shapes. Evaluation SFT judgments project via
#: :func:`project_evaluation`.
SUPPORTED_CAPABILITIES: tuple[str, ...] = tuple(_PROJECTORS)


def project_record(capability: str, record: Mapping[str, Any]) -> ContractProjection:
    """Project ``record`` onto its canonical contract shape for ``capability``.

    Raises:
        ContractAdapterError: if ``capability`` is not one of
            :data:`SUPPORTED_CAPABILITIES`, or the record cannot be projected.
    """
    projector = _PROJECTORS.get(capability)
    if projector is None:
        raise ContractAdapterError(
            f"no contract adapter registered for capability {capability!r}; "
            f"supported: {SUPPORTED_CAPABILITIES}"
        )
    return projector(record)
