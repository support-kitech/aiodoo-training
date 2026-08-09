"""Observation FP2-native generator."""

from __future__ import annotations

from typing import Any

from aiodoo_training.system_training_contract.generators.common import fixture_metadata, rid
from aiodoo_training.system_training_contract.records import ObservationRecord

_SPECS: tuple[dict[str, Any], ...] = (
    {"kind": "execution_result", "status": "succeeded", "capability_id": "workspace.write",
     "summary": "File write observed", "evidence": {"path": "models/partner.py"}, "provider": "execution", "domain": "odoo"},
    {"kind": "execution_result", "status": "failed", "capability_id": "workspace.write",
     "summary": "Write blocked by constraint", "evidence": {"reason": "path_not_writable"}, "provider": "execution", "domain": None},
    {"kind": "validation_result", "status": "failed", "capability_id": "validation.run",
     "summary": "Validation failed", "evidence": {"failed_checks": ["imports"]}, "provider": "execution", "domain": "odoo"},
    {"kind": "validation_result", "status": "succeeded", "capability_id": "validation.run",
     "summary": "Validation passed", "evidence": {"checks_ok": True}, "provider": "execution", "domain": "odoo"},
    {"kind": "repair_result", "status": "succeeded", "capability_id": "execution.repair",
     "summary": "Repair applied", "evidence": {"path": "models/partner.py"}, "provider": "repair", "domain": "odoo"},
    {"kind": "repair_result", "status": "failed", "capability_id": "execution.repair",
     "summary": "Repair unsuccessful", "evidence": {"reason": "insufficient_evidence"}, "provider": "repair", "domain": None},
    {"kind": "artifact_result", "status": "succeeded", "capability_id": "artifact.publish",
     "summary": "Artifact published", "evidence": {"artifact_kind": "patch_bundle"}, "provider": "execution", "domain": None},
    {"kind": "diagnostics_result", "status": "partial", "capability_id": "diagnostics.collect_diagnostics",
     "summary": "Partial diagnostics collected", "evidence": {"coverage": "partial"}, "provider": "execution", "domain": None},
    {"kind": "repository_status", "status": "succeeded", "capability_id": "repository.inspect",
     "summary": "Repository status available", "evidence": {"dirty": True}, "provider": "execution", "domain": None},
    {"kind": "repository_comparison", "status": "succeeded", "capability_id": "repository.compare",
     "summary": "Comparison available", "evidence": {"changed_paths": 2}, "provider": "execution", "domain": None},
    {"kind": "search_result", "status": "succeeded", "capability_id": "workspace.search",
     "summary": "Bounded search matches", "evidence": {"match_count": 3}, "provider": "coding", "domain": "odoo"},
    {"kind": "program_output", "status": "succeeded", "capability_id": "execution.execute_program",
     "summary": "Program completed", "evidence": {"exit_ok": True}, "provider": "execution", "domain": None},
    {"kind": "workspace_change", "status": "succeeded", "capability_id": "workspace.write",
     "summary": "Workspace change observed", "evidence": {"paths": ["a.py"]}, "provider": "execution", "domain": None},
    {"kind": "capability_status", "status": "skipped", "capability_id": "communication.http",
     "summary": "HTTP capability skipped by policy", "evidence": {"reason": "policy_skip"}, "provider": "execution", "domain": None},
    {"kind": "execution_metrics", "status": "succeeded", "capability_id": "execution.execute_program",
     "summary": "Metrics captured", "evidence": {"duration_ms": 12}, "provider": "evaluation", "domain": None},
    {"kind": "generic", "status": "cancelled", "capability_id": "workspace.read",
     "summary": "Observation cancelled", "evidence": {"reason": "user_cancel"}, "provider": "execution", "domain": None},
    {"kind": "environment_status", "status": "succeeded", "capability_id": "workspace.bind",
     "summary": "Workspace bound", "evidence": {"workspace_bound": True}, "provider": "execution", "domain": None},
    {"kind": "repository_history", "status": "succeeded", "capability_id": "repository.history",
     "summary": "History records available", "evidence": {"entry_count": 8}, "provider": "execution", "domain": "odoo"},
    {"kind": "repository_merge", "status": "succeeded", "capability_id": "repository.merge",
     "summary": "Merge completed after approval", "evidence": {"source": "feat/x", "target": "integration"}, "provider": "execution", "domain": None},
    {"kind": "repository_merge", "status": "failed", "capability_id": "repository.merge",
     "summary": "Merge blocked pending approval", "evidence": {"reason": "approval_required"}, "provider": "execution", "domain": None},
    {"kind": "diagnostics_result", "status": "succeeded", "capability_id": "diagnostics.collect_logs",
     "summary": "Bounded logs collected", "evidence": {"bound": "recent", "entries": 12}, "provider": "execution", "domain": None},
    {"kind": "artifact_result", "status": "succeeded", "capability_id": "artifact.attachment",
     "summary": "Artifact attached to context", "evidence": {"attached": True}, "provider": "execution", "domain": None},
    {"kind": "artifact_result", "status": "succeeded", "capability_id": "artifact.import",
     "summary": "Approved artifact imported", "evidence": {"imported": True, "confirmed": True}, "provider": "execution", "domain": "odoo"},
    {"kind": "artifact_result", "status": "failed", "capability_id": "artifact.import",
     "summary": "Import rejected without confirmation", "evidence": {"reason": "confirmation_required"}, "provider": "execution", "domain": None},
)


def generate_observations() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, spec in enumerate(_SPECS, start=1):
        provider = str(spec["provider"])
        domain = spec.get("domain")
        rec = ObservationRecord(
            record_type="observation",
            record_id=rid("obs", i),
            kind=str(spec["kind"]),
            status=str(spec["status"]),
            capability_id=str(spec["capability_id"]),
            summary=str(spec["summary"]),
            evidence=dict(spec.get("evidence") or {}),
            provider_capability=provider,
            domain_specialization=domain if isinstance(domain, str) else None,
            metadata=fixture_metadata(
                generator="observation",
                index=i,
                provider_capability=provider,
                domain_specialization=domain if isinstance(domain, str) else None,
            ),
        )
        out.append(rec.to_dict())
    return out
