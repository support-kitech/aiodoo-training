"""ExecutionWorkUnit FP2-native generator."""

from __future__ import annotations

from typing import Any

from aiodoo_training.system_training_contract.generators.common import fixture_metadata, rid
from aiodoo_training.system_training_contract.records import WorkUnitRecord

_SPECS: tuple[dict[str, Any], ...] = (
    {
        "capability_id": "workspace.write",
        "objective": "Create helper module file",
        "inputs": {"path": "tools/helper.py"},
        "expected_outputs": {"path_exists": True},
        "generated_code": "def ping():\n    return True\n",
        "validation": {"preflight": ["path_writable"]},
        "constraints": {"max_bytes": 4096},
        "provider": "execution",
        "domain": None,
    },
    {
        "capability_id": "workspace.read",
        "objective": "Read existing partner model",
        "inputs": {"path": "models/partner.py"},
        "expected_outputs": {"content_present": True},
        "provider": "coding",
        "domain": "odoo",
    },
    {
        "capability_id": "repository.inspect",
        "objective": "Capture repository status snapshot",
        "inputs": {},
        "expected_outputs": {"status_known": True},
        "provider": "execution",
        "domain": None,
    },
    {
        "capability_id": "repository.compare",
        "objective": "Diff against HEAD",
        "inputs": {"base": "HEAD"},
        "expected_outputs": {"diff_available": True},
        "provider": "execution",
        "domain": None,
    },
    {
        "capability_id": "execution.execute_program",
        "objective": "Execute prepared program WHAT",
        "inputs": {"entrypoint": "tools.helper:ping"},
        "expected_outputs": {"exit_ok": True},
        "validation": {"postflight": ["exit_ok"]},
        "provider": "execution",
        "domain": None,
    },
    {
        "capability_id": "execution.repair",
        "objective": "Repair broken import in partner model",
        "inputs": {"path": "models/partner.py", "symptom": "ImportError"},
        "expected_outputs": {"import_ok": True},
        "provider": "repair",
        "domain": "odoo",
    },
    {
        "capability_id": "validation.run",
        "objective": "Validate module after edits",
        "inputs": {"scope": "module"},
        "expected_outputs": {"validation_ok": True},
        "validation": {"checks": ["imports", "manifest"]},
        "provider": "execution",
        "domain": "odoo",
    },
    {
        "capability_id": "diagnostics.analyze_problems",
        "objective": "Analyze reported problems",
        "inputs": {"problem_ids": ["P1"]},
        "expected_outputs": {"analysis_present": True},
        "provider": "execution",
        "domain": None,
    },
    {
        "capability_id": "artifact.export",
        "objective": "Export change artifact",
        "inputs": {"artifact_kind": "patch_bundle"},
        "expected_outputs": {"artifact_ref": True},
        "generated_artifacts": ("patch_bundle",),
        "provider": "execution",
        "domain": None,
    },
    {
        "capability_id": "communication.http",
        "objective": "Request remote health endpoint",
        "inputs": {"method": "GET", "url": "https://example.invalid/health"},
        "expected_outputs": {"status_class": "2xx"},
        "constraints": {"timeout_ms": 5000},
        "provider": "execution",
        "domain": None,
    },
    {
        "capability_id": "workspace.search",
        "objective": "Search for action_confirm definitions",
        "inputs": {"query": "def action_confirm"},
        "expected_outputs": {"matches_bounded": True},
        "provider": "coding",
        "domain": "odoo",
    },
    {
        "capability_id": "repository.branch",
        "objective": "Create feature branch intent",
        "inputs": {"branch": "feat/partner-field"},
        "expected_outputs": {"branch_ready": True},
        "provider": "execution",
        "domain": None,
    },
    {
        "capability_id": "workspace.bind",
        "objective": "Bind product workspace root before engineering work",
        "inputs": {"root_hint": "project"},
        "expected_outputs": {"workspace_bound": True},
        "provider": "execution",
        "domain": None,
    },
    {
        "capability_id": "repository.history",
        "objective": "Read recent history for changed module paths",
        "inputs": {"path": "models", "limit": 15},
        "expected_outputs": {"history_available": True},
        "provider": "execution",
        "domain": "odoo",
    },
    {
        "capability_id": "repository.merge",
        "objective": "Merge approved feature line into integration",
        "inputs": {"source": "feat/partner-field", "target": "integration", "requires_approval": True},
        "expected_outputs": {"merge_completed": True},
        "constraints": {"requires_approval": True},
        "provider": "execution",
        "domain": None,
    },
    {
        "capability_id": "diagnostics.collect_logs",
        "objective": "Collect bounded session logs after failure",
        "inputs": {"scope": "session", "bound": "recent"},
        "expected_outputs": {"logs_bounded": True},
        "provider": "execution",
        "domain": None,
    },
    {
        "capability_id": "artifact.attachment",
        "objective": "Attach patch artifact to current engineering context",
        "inputs": {"path": "artifacts/change.patch", "artifact_kind": "patch_bundle"},
        "expected_outputs": {"attached": True},
        "provider": "execution",
        "domain": None,
    },
    {
        "capability_id": "artifact.import",
        "objective": "Import approved artifact into workspace",
        "inputs": {"artifact_kind": "patch_bundle", "confirmed": True},
        "expected_outputs": {"imported": True},
        "constraints": {"requires_approval": True},
        "provider": "execution",
        "domain": "odoo",
    },
)


def generate_work_units() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, spec in enumerate(_SPECS, start=1):
        provider = str(spec["provider"])
        domain = spec.get("domain")
        rec = WorkUnitRecord(
            record_type="execution_work_unit",
            record_id=rid("wu", i),
            work_id=f"ewu-fp2-{i:03d}",
            capability_id=str(spec["capability_id"]),
            objective=str(spec["objective"]),
            inputs=dict(spec.get("inputs") or {}),
            expected_outputs=dict(spec.get("expected_outputs") or {}),
            generated_code=str(spec.get("generated_code") or ""),
            generated_artifacts=tuple(spec.get("generated_artifacts") or ()),
            validation=dict(spec.get("validation") or {}),
            constraints=dict(spec.get("constraints") or {}),
            provider_capability=provider,
            domain_specialization=domain if isinstance(domain, str) else None,
            metadata=fixture_metadata(
                generator="execution_work_unit",
                index=i,
                provider_capability=provider,
                domain_specialization=domain if isinstance(domain, str) else None,
            ),
        )
        out.append(rec.to_dict())
    return out
