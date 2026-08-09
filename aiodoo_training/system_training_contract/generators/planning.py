"""Planning Decision FP2-native generator."""

from __future__ import annotations

from typing import Any

from aiodoo_training.system_training_contract.generators.common import fixture_metadata, rid
from aiodoo_training.system_training_contract.records import PlanningDecisionRecord

_SPECS: tuple[dict[str, Any], ...] = (
    {
        "goal": "Add partner computed field and validate",
        "decision_kind": "replan",
        "summary": "Write then validate",
        "steps": (
            {"action": "workspace.read", "args": {"path": "models/partner.py"}},
            {"action": "workspace.write", "args": {"path": "models/partner.py"}},
            {"action": "validation.run", "args": {"scope": "module"}},
        ),
        "domain": "odoo",
    },
    {
        "goal": "Inspect repository before editing",
        "decision_kind": "replan",
        "summary": "Inspect then compare",
        "steps": (
            {"action": "repository.inspect", "args": {}},
            {"action": "repository.compare", "args": {"base": "HEAD"}},
        ),
        "domain": None,
    },
    {
        "goal": "Collect diagnostics for failing objective",
        "decision_kind": "replan",
        "summary": "Diagnostics then analyze",
        "steps": (
            {"action": "diagnostics.collect_diagnostics", "args": {"scope": "workspace"}},
            {"action": "diagnostics.analyze_problems", "args": {}},
        ),
        "domain": None,
    },
    {
        "goal": "Repair import then re-validate",
        "decision_kind": "replan",
        "summary": "Repair then validation.run",
        "steps": (
            {"action": "execution.repair", "args": {"path": "models/partner.py"}},
            {"action": "validation.run", "args": {"scope": "module"}},
        ),
        "domain": "odoo",
    },
    {
        "goal": "Publish artifact after successful validation",
        "decision_kind": "replan",
        "summary": "Validate then publish",
        "steps": (
            {"action": "validation.run", "args": {"scope": "module"}},
            {"action": "artifact.publish", "args": {"artifact_kind": "patch_bundle"}},
        ),
        "domain": None,
    },
    {
        "goal": "Search then read matching file",
        "decision_kind": "replan",
        "summary": "Search navigate read",
        "steps": (
            {"action": "workspace.search", "args": {"query": "action_confirm"}},
            {"action": "workspace.navigate", "args": {"path": "models/sale_order.py"}},
            {"action": "workspace.read", "args": {"path": "models/sale_order.py"}},
        ),
        "domain": "odoo",
    },
    {
        "goal": "Objective already satisfied",
        "decision_kind": "complete",
        "summary": "No further Engineering work",
        "steps": (),
        "domain": None,
    },
    {
        "goal": "Blocked on missing credentials — escalate",
        "decision_kind": "escalate",
        "summary": "Cannot continue without external approval",
        "steps": (),
        "domain": None,
    },
    {
        "goal": "Branch then modify repository",
        "decision_kind": "replan",
        "summary": "Branch and modify",
        "steps": (
            {"action": "repository.branch", "args": {"branch": "feat/x"}},
            {"action": "repository.modify", "args": {"paths": ["a.py"]}},
        ),
        "domain": None,
    },
    {
        "goal": "Export artifact for review",
        "decision_kind": "replan",
        "summary": "Export only",
        "steps": ({"action": "artifact.export", "args": {"artifact_kind": "patch_bundle"}},),
        "domain": None,
    },
    {
        "goal": "Static analysis before write",
        "decision_kind": "replan",
        "summary": "Analyze then write",
        "steps": (
            {"action": "diagnostics.static_analysis", "args": {"scope": "models"}},
            {"action": "workspace.write", "args": {"path": "models/partner.py"}},
        ),
        "domain": "odoo",
    },
    {
        "goal": "HTTP health check then continue planning",
        "decision_kind": "replan",
        "summary": "External health WHAT",
        "steps": (
            {
                "action": "communication.http",
                "args": {"method": "GET", "url": "https://example.invalid/health"},
            },
        ),
        "domain": None,
    },
    {
        "goal": "Bind workspace then inspect repository history",
        "decision_kind": "replan",
        "summary": "Bind and history",
        "steps": (
            {"action": "workspace.bind", "args": {"root_hint": "project"}},
            {"action": "repository.history", "args": {"path": "models", "limit": 10}},
        ),
        "domain": None,
    },
    {
        "goal": "Collect logs then analyze problems",
        "decision_kind": "replan",
        "summary": "Logs then analyze — evidence gathering only",
        "steps": (
            {"action": "diagnostics.collect_logs", "args": {"scope": "session", "bound": "recent"}},
            {"action": "diagnostics.analyze_problems", "args": {}},
        ),
        "domain": None,
    },
    {
        "goal": "Attach artifact then import after approval",
        "decision_kind": "replan",
        "summary": "Attachment then approved import",
        "steps": (
            {"action": "artifact.attachment", "args": {"path": "artifacts/change.patch"}},
            {"action": "artifact.import", "args": {"artifact_kind": "patch_bundle", "confirmed": True}},
        ),
        "domain": "odoo",
    },
    {
        "goal": "Merge feature line after independent approval evidence",
        "decision_kind": "replan",
        "summary": "Merge WHAT — approval is separate decision",
        "steps": (
            {"action": "repository.compare", "args": {"base": "integration"}},
            {"action": "repository.merge", "args": {"source": "feat/partner-field", "target": "integration"}},
        ),
        "domain": None,
    },
)


def generate_planning_decisions() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, spec in enumerate(_SPECS, start=1):
        domain = spec.get("domain")
        rec = PlanningDecisionRecord(
            record_type="planning_decision",
            record_id=rid("pd", i),
            goal=str(spec["goal"]),
            decision_kind=str(spec["decision_kind"]),
            summary=str(spec["summary"]),
            steps=tuple(spec["steps"]),
            provider_capability="planner",
            domain_specialization=domain if isinstance(domain, str) else None,
            metadata=fixture_metadata(
                generator="planning_decision",
                index=i,
                provider_capability="planner",
                domain_specialization=domain if isinstance(domain, str) else None,
            ),
        )
        out.append(rec.to_dict())
    return out
