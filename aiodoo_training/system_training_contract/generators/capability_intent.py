"""Capability Intent FP2-native generator."""

from __future__ import annotations

from typing import Any

from aiodoo_training.system_training_contract.generators.common import fixture_metadata, rid
from aiodoo_training.system_training_contract.records import CapabilityIntentRecord

# (capability_id, objective, args, provider, domain)
_SPECS: tuple[tuple[str, str, dict[str, Any], str, str | None], ...] = (
    ("workspace.bind", "Bind active engineering workspace", {"root_hint": "project"}, "execution", None),
    ("workspace.bind", "Bind module workspace for Odoo engineering session", {"root_hint": "addons/sale"}, "execution", "odoo"),
    ("workspace.read", "Read partner model source", {"path": "models/partner.py"}, "coding", "odoo"),
    ("workspace.write", "Add computed field stub", {"path": "models/partner.py"}, "coding", "odoo"),
    ("workspace.search", "Locate sale order confirm methods", {"query": "action_confirm"}, "coding", "odoo"),
    ("workspace.navigate", "Open module manifest", {"path": "__manifest__.py"}, "coding", "odoo"),
    ("repository.inspect", "Inspect repository status", {}, "execution", None),
    ("repository.compare", "Compare working tree to HEAD", {"base": "HEAD"}, "execution", None),
    ("repository.history", "Show recent commits for models/", {"path": "models"}, "execution", None),
    ("repository.history", "Review history for partner model changes", {"path": "models/partner.py", "limit": 20}, "execution", "odoo"),
    ("repository.merge", "Merge feature line of work into integration line", {"source": "feat/partner-field", "target": "integration"}, "execution", None),
    ("repository.modify", "Stage intended repository change set", {"paths": ["models/partner.py"]}, "execution", None),
    ("execution.execute_program", "Run module entrypoint under constraints", {"entrypoint": "main"}, "execution", None),
    ("execution.repair", "Repair failing import after edit", {"path": "models/partner.py"}, "repair", "odoo"),
    ("communication.http", "Fetch remote API health status", {"url": "https://example.invalid/health"}, "execution", None),
    ("diagnostics.collect_diagnostics", "Collect structured diagnostics", {"scope": "workspace"}, "execution", None),
    ("diagnostics.collect_logs", "Collect bounded log evidence for failing objective", {"scope": "session", "bound": "recent"}, "execution", None),
    ("diagnostics.static_analysis", "Run static analysis WHAT", {"scope": "models"}, "execution", "odoo"),
    ("artifact.attachment", "Bind published patch artifact to engineering context", {"artifact_kind": "patch_bundle", "path": "artifacts/change.patch"}, "execution", None),
    ("artifact.publish", "Publish generated change artifact", {"artifact_kind": "patch_bundle"}, "execution", None),
    ("artifact.import", "Import approved artifact into workspace", {"artifact_kind": "patch_bundle", "confirmed": True}, "execution", None),
    ("validation.run", "Validate objective completion criteria", {"suite": "module"}, "execution", "odoo"),
)


def generate_capability_intents() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, (cap, objective, args, provider, domain) in enumerate(_SPECS, start=1):
        rec = CapabilityIntentRecord(
            record_type="capability_intent",
            record_id=rid("ci", i),
            capability_id=cap,
            objective=objective,
            args=args,
            reason="fp2-native fixture",
            provider_capability=provider,
            domain_specialization=domain,
            metadata=fixture_metadata(
                generator="capability_intent",
                index=i,
                provider_capability=provider,
                domain_specialization=domain,
            ),
        )
        out.append(rec.to_dict())
    return out
