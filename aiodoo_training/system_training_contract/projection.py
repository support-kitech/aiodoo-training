"""Historical → Canonical Training projection layer (TR-2).

Preserves provenance. Never silently fabricates Engineering capability IDs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import uuid4

from aiodoo_training.system_training_contract.taxonomy import (
    DEVELOPMENT_PROVIDER_CAPABILITIES,
    PREFERRED_ENGINEERING_CAPABILITY_IDS,
    PROVIDER_CAPABILITY_IDS,
    REASONING_PROVIDER_CAPABILITIES,
    TaxonomyPlane,
    classify_capability_id,
)
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

__all__ = [
    "ProjectionStatus",
    "Provenance",
    "ProjectionResult",
    "project_historical_record",
    "PROJECTION_VERSION",
]

PROJECTION_VERSION: str = "1.0.0"

# Historical Protocol V1 / Agent-era planner actions — NOT proven equivalent to
# preferred Engineering IDs. Mark unsupported unless explicitly allowlisted later.
_HISTORICAL_PLANNER_ACTIONS: frozenset[str] = frozenset(
    {
        "create_file",
        "modify_file",
        "update_file",
        "delete_file",
        "read_file",
        "mkdir",
        "run_command",
        "apply_patch",
        "apply_artifact",
        "write_file",
        "edit_file",
        "search_code",
        "git_commit",
        "git_checkout",
        "shell",
        "bash",
    }
)

# Lossy execution-dataset markers (TR-1).
_LOSSY_EXECUTION_MARKERS: frozenset[str] = frozenset(
    {
        "apply_artifact",
        "artifact_apply",
        "apply_diff",
        "write_artifact",
        "local_artifact",
    }
)


class ProjectionStatus(StrEnum):
    PROJECTED = "projected"
    PARTIALLY_PROJECTED = "partially_projected"
    UNSUPPORTED = "unsupported"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Provenance:
    source_dataset: str
    source_record_id: str
    source_schema_version: str = "legacy"
    projection_version: str = PROJECTION_VERSION
    projection_status: ProjectionStatus = ProjectionStatus.UNSUPPORTED
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_dataset": self.source_dataset,
            "source_record_id": self.source_record_id,
            "source_schema_version": self.source_schema_version,
            "projection_version": self.projection_version,
            "projection_status": self.projection_status.value,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    status: ProjectionStatus
    provenance: Provenance
    canonical: Mapping[str, Any] | None = None
    reasons: tuple[str, ...] = ()
    provider_capability: str | None = None
    domain_specialization: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "provenance": self.provenance.to_dict(),
            "canonical": dict(self.canonical) if self.canonical else None,
            "reasons": list(self.reasons),
            "provider_capability": self.provider_capability,
            "domain_specialization": self.domain_specialization,
        }


def _detect_provider_capability(
    *,
    dataset: str,
    record: Mapping[str, Any],
) -> str | None:
    explicit = str(
        record.get("provider_capability")
        or record.get("capability")
        or record.get("adapter_capability")
        or ""
    ).strip().lower()
    if explicit in PROVIDER_CAPABILITY_IDS:
        return explicit
    ds = dataset.strip().lower()
    for cap in PROVIDER_CAPABILITY_IDS:
        if cap in ds:
            return cap
    return None


def _detect_odoo(record: Mapping[str, Any], dataset: str) -> str | None:
    blob = " ".join(
        [
            dataset.lower(),
            str(record.get("domain") or "").lower(),
            str(record.get("domain_specialization") or "").lower(),
            str(record.get("source") or "").lower(),
        ]
    )
    if "odoo" in blob:
        return "odoo"
    return None


def _reject(prov: Provenance, *reasons: str) -> ProjectionResult:
    return ProjectionResult(
        status=ProjectionStatus.REJECTED,
        provenance=Provenance(
            source_dataset=prov.source_dataset,
            source_record_id=prov.source_record_id,
            source_schema_version=prov.source_schema_version,
            projection_version=prov.projection_version,
            projection_status=ProjectionStatus.REJECTED,
            notes="; ".join(reasons),
        ),
        reasons=reasons,
    )


def _unsupported(prov: Provenance, *reasons: str, provider: str | None = None, domain: str | None = None) -> ProjectionResult:
    return ProjectionResult(
        status=ProjectionStatus.UNSUPPORTED,
        provenance=Provenance(
            source_dataset=prov.source_dataset,
            source_record_id=prov.source_record_id,
            source_schema_version=prov.source_schema_version,
            projection_version=prov.projection_version,
            projection_status=ProjectionStatus.UNSUPPORTED,
            notes="; ".join(reasons),
        ),
        reasons=reasons,
        provider_capability=provider,
        domain_specialization=domain,
    )


def project_historical_record(
    record: Mapping[str, Any],
    *,
    source_dataset: str,
    source_record_id: str | None = None,
    source_schema_version: str = "legacy",
) -> ProjectionResult:
    """Project one historical Training/dataset record into a canonical form.

    Rules (TR-2):
    - Do not fabricate Engineering capability IDs from Protocol V1 actions.
    - Keep provider-capability records as provider-plane (not Engineering WU).
    - Mark lossy execution / unsupported planner actions as UNSUPPORTED.
    - Preserve Odoo domain specialization labels without claiming genericity.
    """
    if not isinstance(record, Mapping):
        rid = source_record_id or "unknown"
        return _reject(
            Provenance(
                source_dataset=source_dataset,
                source_record_id=rid,
                source_schema_version=source_schema_version,
            ),
            "record must be a mapping",
        )

    rid = str(
        source_record_id
        or record.get("record_id")
        or record.get("id")
        or f"hist-{uuid4().hex[:12]}"
    )
    prov = Provenance(
        source_dataset=source_dataset,
        source_record_id=rid,
        source_schema_version=source_schema_version,
    )
    provider = _detect_provider_capability(dataset=source_dataset, record=record)
    domain = _detect_odoo(record, source_dataset)

    # Already canonical?
    if str(record.get("training_contract_version") or "") == SYSTEM_TRAINING_CONTRACT_VERSION:
        rtype = str(record.get("record_type") or "")
        if rtype:
            return ProjectionResult(
                status=ProjectionStatus.PROJECTED,
                provenance=Provenance(
                    source_dataset=source_dataset,
                    source_record_id=rid,
                    source_schema_version=source_schema_version,
                    projection_status=ProjectionStatus.PROJECTED,
                    notes="already canonical",
                ),
                canonical=dict(record),
                reasons=("already_canonical",),
                provider_capability=provider or (
                    str(record.get("provider_capability") or "") or None
                ),
                domain_specialization=domain or (
                    str(record.get("domain_specialization") or "") or None
                ),
            )

    action = ""
    for key in ("action", "capability_id"):
        if record.get(key):
            action = str(record.get(key)).strip()
            break
    if not action:
        steps_raw = record.get("steps")
        if isinstance(steps_raw, list) and steps_raw and isinstance(steps_raw[0], Mapping):
            action = str(steps_raw[0].get("action") or steps_raw[0].get("capability_id") or "").strip()
    if action and classify_capability_id(action) is TaxonomyPlane.FORBIDDEN_HOW:
        return _reject(prov, f"forbidden HOW capability/action: {action!r}")

    # Planner historical actions
    if provider == "planner" or "planner" in source_dataset.lower():
        steps = record.get("steps") or record.get("plan") or record.get("actions") or []
        if isinstance(steps, list) and steps:
            for step in steps:
                if not isinstance(step, Mapping):
                    continue
                act = str(step.get("action") or step.get("capability_id") or "").strip().lower()
                if act in _HISTORICAL_PLANNER_ACTIONS or classify_capability_id(act) is TaxonomyPlane.UNKNOWN:
                    if act not in {""} and classify_capability_id(act) is not TaxonomyPlane.ENGINEERING:
                        return _unsupported(
                            prov,
                            f"historical planner action {act!r} has no proven Engineering mapping",
                            provider=provider,
                            domain=domain,
                        )
        # Single-action planner rows
        if action.lower() in _HISTORICAL_PLANNER_ACTIONS:
            return _unsupported(
                prov,
                f"historical planner action {action!r} has no proven Engineering mapping",
                provider=provider,
                domain=domain,
            )

    # Execution dataset lossy semantics
    if provider == "execution" or "execution" in source_dataset.lower():
        text = str(record).lower()
        if any(m in text for m in _LOSSY_EXECUTION_MARKERS):
            return _unsupported(
                prov,
                "execution record uses lossy artifact/apply-style semantics; FP2-native WorkUnit required",
                provider=provider or "execution",
                domain=domain,
            )
        # Even without markers, historical execution is not proven FP2 WorkUnit.
        if not record.get("capability_id") or classify_capability_id(
            str(record.get("capability_id") or "")
        ) is not TaxonomyPlane.ENGINEERING:
            return _unsupported(
                prov,
                "execution record lacks Engineering capability_id / FP2 WorkUnit fields",
                provider=provider or "execution",
                domain=domain,
            )

    # Coding / repair / context — keep as provider-plane specialization, do not
    # auto-convert to Engineering Work Units.
    if provider in DEVELOPMENT_PROVIDER_CAPABILITIES - {"execution"}:
        if provider == "repair":
            # Only map to execution.repair when explicitly equivalent.
            target = str(record.get("capability_id") or record.get("target_capability") or "").strip()
            if target == "execution.repair":
                canonical = {
                    "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
                    "record_type": "capability_intent",
                    "record_id": f"proj-{rid}",
                    "system_contract": "execution.capability_intent",
                    "provider_capability": "repair",
                    "domain_specialization": domain,
                    "provenance": Provenance(
                        source_dataset=source_dataset,
                        source_record_id=rid,
                        source_schema_version=source_schema_version,
                        projection_status=ProjectionStatus.PROJECTED,
                        notes="repair → execution.repair (explicit)",
                    ).to_dict(),
                    "metadata": {"historical_provider_record": True},
                    "input": {
                        "objective": str(record.get("objective") or record.get("goal") or "repair"),
                        "reason": "projected from historical repair record",
                    },
                    "expected_output": {
                        "capability_id": "execution.repair",
                        "args": dict(record.get("args") or {}),
                    },
                }
                return ProjectionResult(
                    status=ProjectionStatus.PROJECTED,
                    provenance=Provenance(
                        source_dataset=source_dataset,
                        source_record_id=rid,
                        source_schema_version=source_schema_version,
                        projection_status=ProjectionStatus.PROJECTED,
                        notes="repair → execution.repair",
                    ),
                    canonical=canonical,
                    reasons=("explicit_execution_repair",),
                    provider_capability="repair",
                    domain_specialization=domain,
                )
            return ProjectionResult(
                status=ProjectionStatus.PARTIALLY_PROJECTED,
                provenance=Provenance(
                    source_dataset=source_dataset,
                    source_record_id=rid,
                    source_schema_version=source_schema_version,
                    projection_status=ProjectionStatus.PARTIALLY_PROJECTED,
                    notes="provider repair pack; not auto-mapped to Engineering",
                ),
                canonical={
                    "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
                    "projection_envelope": True,
                    "system_contract": "training.provider_capability",
                    "provider_capability": "repair",
                    "domain_specialization": domain,
                    "provenance": Provenance(
                        source_dataset=source_dataset,
                        source_record_id=rid,
                        source_schema_version=source_schema_version,
                        projection_status=ProjectionStatus.PARTIALLY_PROJECTED,
                        notes="preserve as provider-plane specialization",
                    ).to_dict(),
                    "metadata": {
                        "plane": TaxonomyPlane.PROVIDER.value,
                        "historical_payload_keys": sorted(str(k) for k in record.keys()),
                        "do_not_auto_convert_to_work_unit": True,
                    },
                },
                reasons=("provider_repair_preserved", "no_automatic_engineering_map"),
                provider_capability="repair",
                domain_specialization=domain,
            )

        # coding / context — preserve provider plane only
        return ProjectionResult(
            status=ProjectionStatus.PARTIALLY_PROJECTED,
            provenance=Provenance(
                source_dataset=source_dataset,
                source_record_id=rid,
                source_schema_version=source_schema_version,
                projection_status=ProjectionStatus.PARTIALLY_PROJECTED,
                notes=f"preserve {provider} as provider-plane specialization",
            ),
            canonical={
                "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
                "projection_envelope": True,
                "system_contract": "training.provider_capability",
                "provider_capability": provider,
                "domain_specialization": domain,
                "provenance": Provenance(
                    source_dataset=source_dataset,
                    source_record_id=rid,
                    source_schema_version=source_schema_version,
                    projection_status=ProjectionStatus.PARTIALLY_PROJECTED,
                    notes=f"provider {provider} not converted to Engineering WorkUnit",
                ).to_dict(),
                "metadata": {
                    "plane": TaxonomyPlane.PROVIDER.value,
                    "do_not_auto_convert_to_work_unit": True,
                    "historical_payload_keys": sorted(str(k) for k in record.keys()),
                },
            },
            reasons=(f"provider_{provider}_preserved", "no_automatic_engineering_map"),
            provider_capability=provider,
            domain_specialization=domain,
        )

    # Reasoning packs (conversation / approval / evaluation)
    if provider in REASONING_PROVIDER_CAPABILITIES - {"planner"}:
        # Map only loosely to loop/decision surfaces when decision_kind present.
        kind = str(record.get("decision_kind") or record.get("decision") or "").strip().lower()
        if provider == "approval" and kind in {"approve", "reject", "modify"}:
            canonical = {
                "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
                "record_type": "loop_decision",
                "record_id": f"proj-{rid}",
                "system_contract": "intelligence_loop.decision",
                "provider_capability": "approval",
                "domain_specialization": domain,
                "provenance": Provenance(
                    source_dataset=source_dataset,
                    source_record_id=rid,
                    source_schema_version=source_schema_version,
                    projection_status=ProjectionStatus.PROJECTED,
                    notes="approval decision → loop_decision",
                ).to_dict(),
                "metadata": {},
                "expected_output": {
                    "decision_kind": kind,
                    "reason": str(record.get("reason") or "approval decision"),
                    "next_goal": str(record.get("next_goal") or ""),
                },
            }
            return ProjectionResult(
                status=ProjectionStatus.PROJECTED,
                provenance=Provenance(
                    source_dataset=source_dataset,
                    source_record_id=rid,
                    source_schema_version=source_schema_version,
                    projection_status=ProjectionStatus.PROJECTED,
                    notes="approval → loop_decision",
                ),
                canonical=canonical,
                reasons=("approval_loop_decision",),
                provider_capability="approval",
                domain_specialization=domain,
            )
        return _unsupported(
            prov,
            f"{provider} historical record has no proven DecisionContext/loop mapping",
            provider=provider,
            domain=domain,
        )

    # Explicit preferred Engineering capability already present
    eng_id = str(record.get("capability_id") or "").strip()
    if eng_id and eng_id in PREFERRED_ENGINEERING_CAPABILITY_IDS:
        canonical = {
            "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
            "record_type": "capability_intent",
            "record_id": f"proj-{rid}",
            "system_contract": "execution.capability_intent",
            "provider_capability": provider,
            "domain_specialization": domain,
            "provenance": Provenance(
                source_dataset=source_dataset,
                source_record_id=rid,
                source_schema_version=source_schema_version,
                projection_status=ProjectionStatus.PROJECTED,
                notes="historical record already carried Engineering capability_id",
            ).to_dict(),
            "metadata": {},
            "input": {
                "objective": str(record.get("objective") or record.get("goal") or eng_id),
                "reason": "projected from historical Engineering-tagged record",
            },
            "expected_output": {
                "capability_id": eng_id,
                "args": dict(record.get("args") or {}),
            },
        }
        return ProjectionResult(
            status=ProjectionStatus.PROJECTED,
            provenance=Provenance(
                source_dataset=source_dataset,
                source_record_id=rid,
                source_schema_version=source_schema_version,
                projection_status=ProjectionStatus.PROJECTED,
                notes="Engineering capability_id present",
            ),
            canonical=canonical,
            reasons=("engineering_capability_present",),
            provider_capability=provider,
            domain_specialization=domain,
        )

    return _unsupported(
        prov,
        "no proven projection rule for historical record shape",
        provider=provider,
        domain=domain,
    )
