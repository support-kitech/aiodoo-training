"""FP2-native Context provider fixtures (AT-6.2).

Context = Development retrieval / repository-locate specialization
(System: DEVELOPMENT_CAPABILITIES includes \"context\"; Training Contract:
\"Odoo retrieval specialization → preserve; not Work Units\").

Emits canonical Training Contract records with provider_capability=context.
Does NOT project legacy context_v1_0.jsonl.
Does NOT modify controlled_batch_2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aiodoo_training.system_training_contract.generators.common import (
    fixture_metadata,
    rid,
    write_jsonl,
)
from aiodoo_training.system_training_contract.generators.mapping import (
    CONTEXT_ALLOWED_RECORD_TYPES,
)
from aiodoo_training.system_training_contract.records import (
    CapabilityIntentRecord,
    ObservationRecord,
    validate_record_mapping,
)
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

CONTEXT_CORPUS_VERSION: str = "fp2-context-1.0.0"
CONTEXT_GENERATOR_NAME: str = "context"

_REPO_ROOT = Path(__file__).resolve().parents[3]  # aiodoo-training/
_WORKSPACE_ROOT = _REPO_ROOT.parent
_DEFAULT_TRAINING_FIXTURES = _REPO_ROOT / "fixtures" / "fp2" / "context"
_DEFAULT_DATASETS = _WORKSPACE_ROOT / "aiodoo-datasets" / "datasets" / "fp2" / "context"

# Locate / retrieve Engineering WHAT only (preferred IDs).
# Inspired by legacy retrieval intents (find_model, locate_field, …) — not projected.
_INTENT_SPECS: tuple[tuple[str, str, dict[str, Any], str | None], ...] = (
    ("workspace.search", "Locate definition of res.partner model", {"query": "res.partner"}, "odoo"),
    ("workspace.search", "Find AuthTotpDevice class definition", {"query": "AuthTotpDevice"}, "odoo"),
    ("workspace.search", "Locate sale.order confirm method", {"query": "action_confirm"}, "odoo"),
    ("repository.inspect", "Locate repository module layout for sale", {"root_hint": "addons/sale"}, "odoo"),
    ("workspace.navigate", "Navigate to partner model source path", {"path": "models/partner.py"}, "odoo"),
    ("workspace.read", "Open located partner model for inspection", {"path": "models/partner.py"}, "odoo"),
    ("workspace.search", "Find field vat on partner model", {"query": "vat", "scope": "models/partner.py"}, "odoo"),
    ("workspace.search", "Locate __manifest__ for sale module", {"query": "__manifest__.py", "scope": "addons/sale"}, "odoo"),
    ("repository.inspect", "Inspect repository tree for models package", {"path": "models"}, None),
    ("workspace.search", "Find HTTP controller route definition", {"query": "type='http'"}, None),
    ("workspace.navigate", "Navigate to controllers package", {"path": "controllers"}, None),
    ("workspace.search", "Locate security access CSV for module", {"query": "ir.model.access.csv"}, "odoo"),
    ("workspace.search", "Locate view architecture for sale.order form", {"query": "sale.order.form"}, "odoo"),
    ("workspace.search", "Find related field definition on sale.order", {"query": "partner_id", "scope": "models/sale_order.py"}, "odoo"),
    ("workspace.read", "Read located controller after retrieval", {"path": "controllers/main.py"}, None),
    ("repository.inspect", "Inspect addons root for module discovery", {"path": "addons"}, "odoo"),
)

_OBS_SPECS: tuple[dict[str, Any], ...] = (
    {
        "kind": "search_result",
        "status": "succeeded",
        "capability_id": "workspace.search",
        "summary": "Located model definition for AuthTotpDevice",
        "evidence": {
            "match_count": 1,
            "ranked_artifacts": [
                {
                    "path": "models/auth_totp_device.py",
                    "symbol": "AuthTotpDevice",
                    "artifact_kind": "model",
                    "score": 100,
                    "ranking_reason": "direct_definition",
                }
            ],
        },
        "domain": "odoo",
    },
    {
        "kind": "search_result",
        "status": "succeeded",
        "capability_id": "workspace.search",
        "summary": "Located field vat on partner model",
        "evidence": {
            "match_count": 2,
            "ranked_artifacts": [
                {
                    "path": "models/partner.py",
                    "symbol": "vat",
                    "artifact_kind": "field",
                    "score": 95,
                    "ranking_reason": "field_definition",
                }
            ],
        },
        "domain": "odoo",
    },
    {
        "kind": "search_result",
        "status": "partial",
        "capability_id": "workspace.search",
        "summary": "Ambiguous search for confirm — multiple matches",
        "evidence": {"match_count": 8, "ambiguous": True},
        "domain": "odoo",
    },
    {
        "kind": "search_result",
        "status": "failed",
        "capability_id": "workspace.search",
        "summary": "No matches for unknown symbol",
        "evidence": {"match_count": 0, "query": "TotallyMissingModel"},
        "domain": None,
    },
    {
        "kind": "repository_status",
        "status": "succeeded",
        "capability_id": "repository.inspect",
        "summary": "Repository layout available for locate",
        "evidence": {"models_present": True, "path": "models"},
        "domain": None,
    },
    {
        "kind": "search_result",
        "status": "succeeded",
        "capability_id": "workspace.search",
        "summary": "Located sale.order action_confirm",
        "evidence": {
            "match_count": 1,
            "ranked_artifacts": [
                {
                    "path": "models/sale_order.py",
                    "symbol": "action_confirm",
                    "artifact_kind": "method",
                    "score": 98,
                    "ranking_reason": "method_definition",
                }
            ],
        },
        "domain": "odoo",
    },
    {
        "kind": "workspace_change",
        "status": "succeeded",
        "capability_id": "workspace.navigate",
        "summary": "Navigated to retrieved path controllers/main.py",
        "evidence": {"path": "controllers/main.py", "opened": True},
        "domain": None,
    },
    {
        "kind": "search_result",
        "status": "succeeded",
        "capability_id": "workspace.search",
        "summary": "Located module manifest",
        "evidence": {
            "match_count": 1,
            "ranked_artifacts": [
                {
                    "path": "addons/sale/__manifest__.py",
                    "symbol": "__manifest__",
                    "artifact_kind": "manifest",
                    "score": 100,
                    "ranking_reason": "direct_path",
                }
            ],
        },
        "domain": "odoo",
    },
    {
        "kind": "search_result",
        "status": "succeeded",
        "capability_id": "workspace.search",
        "summary": "Located sale.order form view",
        "evidence": {
            "match_count": 1,
            "ranked_artifacts": [
                {
                    "path": "views/sale_views.xml",
                    "symbol": "sale.order.form",
                    "artifact_kind": "view",
                    "score": 97,
                    "ranking_reason": "view_id",
                }
            ],
        },
        "domain": "odoo",
    },
    {
        "kind": "artifact_result",
        "status": "succeeded",
        "capability_id": "workspace.read",
        "summary": "Retrieved artifact content after locate",
        "evidence": {"path": "models/partner.py", "bytes_read": 4096},
        "domain": "odoo",
    },
)


def _extra(index: int, scenario: str) -> dict[str, Any]:
    return {
        "context_corpus_version": CONTEXT_CORPUS_VERSION,
        "context_scenario": scenario,
        "fp2_native": True,
        "legacy": False,
    }


def generate_context_capability_intents() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, (cap, objective, args, domain) in enumerate(_INTENT_SPECS, start=1):
        rec = CapabilityIntentRecord(
            record_type="capability_intent",
            record_id=rid("ctx-ci", i),
            capability_id=cap,
            objective=objective,
            args=args,
            reason="fp2-context-native fixture",
            provider_capability="context",
            domain_specialization=domain,
            metadata=fixture_metadata(
                generator=CONTEXT_GENERATOR_NAME,
                index=i,
                provider_capability="context",
                domain_specialization=domain,
                extra=_extra(i, "locate_intent"),
            ),
        )
        out.append(rec.to_dict())
    return out


def generate_context_observations() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, spec in enumerate(_OBS_SPECS, start=1):
        domain = spec.get("domain")
        rec = ObservationRecord(
            record_type="observation",
            record_id=rid("ctx-obs", i),
            kind=str(spec["kind"]),
            status=str(spec["status"]),
            capability_id=str(spec["capability_id"]),
            summary=str(spec["summary"]),
            evidence=dict(spec.get("evidence") or {}),
            provider_capability="context",
            domain_specialization=domain if isinstance(domain, str) else None,
            metadata=fixture_metadata(
                generator=CONTEXT_GENERATOR_NAME,
                index=200 + i,
                provider_capability="context",
                domain_specialization=domain if isinstance(domain, str) else None,
                extra=_extra(200 + i, "locate_observation"),
            ),
        )
        out.append(rec.to_dict())
    return out


def generate_context_records() -> list[dict[str, Any]]:
    """All Context-native fixtures (mixed record types, provider=context)."""
    records = generate_context_capability_intents() + generate_context_observations()
    for rec in records:
        validate_record_mapping(rec)
        assert rec["provider_capability"] == "context"
        assert rec["record_type"] in CONTEXT_ALLOWED_RECORD_TYPES
        assert rec["training_contract_version"] == SYSTEM_TRAINING_CONTRACT_VERSION
    return records


def emit_context_fixtures(
    *,
    training_fixtures_root: Path | None = None,
    datasets_root: Path | None = None,
) -> dict[str, Any]:
    """Write versioned Context fixtures; never touches controlled_batch_2."""
    records = generate_context_records()
    by_type: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_type.setdefault(str(rec["record_type"]), []).append(rec)

    training_root = Path(training_fixtures_root or _DEFAULT_TRAINING_FIXTURES)
    datasets_out = Path(datasets_root or _DEFAULT_DATASETS)

    written: dict[str, str] = {}
    for root in (training_root, datasets_out):
        root.mkdir(parents=True, exist_ok=True)
        all_path = root / "context_native.jsonl"
        write_jsonl(all_path, records)
        written[str(all_path)] = f"{len(records)} records"
        for rtype, rows in sorted(by_type.items()):
            p = root / f"{rtype}.jsonl"
            write_jsonl(p, rows)
            written[str(p)] = f"{len(rows)} records"

        manifest = {
            "version": CONTEXT_CORPUS_VERSION,
            "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
            "provider_capability": "context",
            "product_plane": "development",
            "generator": CONTEXT_GENERATOR_NAME,
            "total_records": len(records),
            "by_record_type": {k: len(v) for k, v in sorted(by_type.items())},
            "legacy_projection": False,
            "controlled_batch_2_modified": False,
            "notes": [
                "AT-6.2 Context coverage fixtures only",
                "Not a production training pack",
                "Legacy context_v1_0.jsonl not included",
            ],
        }
        man_path = root / "manifest.json"
        man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written[str(man_path)] = "manifest"

    return {
        "version": CONTEXT_CORPUS_VERSION,
        "count": len(records),
        "written": written,
        "by_type": {k: len(v) for k, v in by_type.items()},
    }
