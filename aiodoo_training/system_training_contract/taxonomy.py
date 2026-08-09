"""Provider vs Engineering taxonomy separation (TR-2).

Two planes — never merge:

1. Provider / adapter capabilities — identify LoRA specialization packs.
2. Engineering capabilities — model-facing Execution WHAT (domain.intent).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "TaxonomyPlane",
    "PROVIDER_CAPABILITY_IDS",
    "DEVELOPMENT_PROVIDER_CAPABILITIES",
    "REASONING_PROVIDER_CAPABILITIES",
    "ENGINEERING_CAPABILITY_IDS",
    "PREFERRED_ENGINEERING_CAPABILITY_IDS",
    "classify_capability_id",
]


class TaxonomyPlane(StrEnum):
    PROVIDER = "provider"
    ENGINEERING = "engineering"
    UNKNOWN = "unknown"
    FORBIDDEN_HOW = "forbidden_how"


DEVELOPMENT_PROVIDER_CAPABILITIES: frozenset[str] = frozenset(
    {"coding", "repair", "execution", "context"}
)
REASONING_PROVIDER_CAPABILITIES: frozenset[str] = frozenset(
    {"planner", "conversation", "approval", "evaluation"}
)
PROVIDER_CAPABILITY_IDS: frozenset[str] = (
    DEVELOPMENT_PROVIDER_CAPABILITIES | REASONING_PROVIDER_CAPABILITIES
)

PREFERRED_ENGINEERING_CAPABILITY_IDS: frozenset[str] = frozenset(
    {
        "workspace.bind",
        "workspace.read",
        "workspace.write",
        "workspace.search",
        "workspace.navigate",
        "repository.inspect",
        "repository.compare",
        "repository.history",
        "repository.branch",
        "repository.merge",
        "repository.modify",
        "execution.execute_program",
        "execution.repair",
        "communication.http",
        "diagnostics.collect_diagnostics",
        "diagnostics.collect_logs",
        "diagnostics.analyze_problems",
        "diagnostics.static_analysis",
        "artifact.attachment",
        "artifact.publish",
        "artifact.export",
        "artifact.import",
        "validation.run",
    }
)

ENGINEERING_CAPABILITY_IDS: frozenset[str] = PREFERRED_ENGINEERING_CAPABILITY_IDS | frozenset(
    {
        "search",
        "read",
        "edit",
        "patch",
        "shell",
        "python.exec",
        "http.request",
        "diagnostics.collect",
        "validate",
        "execution.validate",
        "execution.execute_tests",
        "artifacts.publish",
        "attachments.bind",
    }
)


def classify_capability_id(capability_id: str) -> TaxonomyPlane:
    key = (capability_id or "").strip().lower()
    if not key:
        return TaxonomyPlane.UNKNOWN
    if key.startswith("local_") or key in {
        "local_workspace",
        "local_git",
        "local_program",
        "local_validation",
        "local_artifact",
        "local_diagnostics",
        "local_http",
        "local_snapshot",
        "local_repair",
    }:
        return TaxonomyPlane.FORBIDDEN_HOW
    if key in PROVIDER_CAPABILITY_IDS:
        return TaxonomyPlane.PROVIDER
    if key in ENGINEERING_CAPABILITY_IDS:
        return TaxonomyPlane.ENGINEERING
    return TaxonomyPlane.UNKNOWN
