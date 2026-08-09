"""Forbidden HOW and taxonomy quality gates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aiodoo_training.system_training_contract.forbidden import (
    FORBIDDEN_ARG_KEYS,
    FORBIDDEN_BACKEND_ACTIONS,
    FORBIDDEN_IMPL_IDS,
    ForbiddenHowError,
    assert_no_forbidden_how,
)
from aiodoo_training.system_training_contract.quality.common import (
    extract_engineering_capability,
)
from aiodoo_training.system_training_contract.taxonomy import (
    PREFERRED_ENGINEERING_CAPABILITY_IDS,
    PROVIDER_CAPABILITY_IDS,
    TaxonomyPlane,
    classify_capability_id,
)

# Long / unambiguous HOW tokens scanned in model-facing text.
_TEXT_HOW_TOKENS: frozenset[str] = frozenset(
    {
        *FORBIDDEN_IMPL_IDS,
        "adapters_required",
        "implementation_id",
        "strategy_id",
        "resolver_id",
        "pytest",
        "mypy",
        "pyright",
        "huggingface.co",
        "local_workspace",
        "local_git",
        "local_program",
        "local_validation",
        "local_artifact",
        "local_diagnostics",
    }
)


def _collect_args(record: Mapping[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    expected = record.get("expected_output") if isinstance(record.get("expected_output"), Mapping) else {}
    inp = record.get("input") if isinstance(record.get("input"), Mapping) else {}
    evidence = record.get("evidence") if isinstance(record.get("evidence"), Mapping) else {}
    if isinstance(expected.get("args"), Mapping):
        args.update(expected["args"])
    if isinstance(inp.get("inputs"), Mapping):
        args.update(inp["inputs"])
    if isinstance(inp.get("constraints"), Mapping):
        args.update(inp["constraints"])
    if isinstance(evidence.get("details"), Mapping):
        args.update(evidence["details"])
    # Planning steps
    for step in expected.get("steps") or []:
        if isinstance(step, Mapping) and isinstance(step.get("args"), Mapping):
            args.update(step["args"])
    return args


def _model_facing_text(record: Mapping[str, Any]) -> str:
    parts = [record.get("input"), record.get("expected_output"), record.get("evidence")]
    return str(parts).lower()


def scan_forbidden_how(record: Mapping[str, Any]) -> list[str]:
    """Return issues for model-facing HOW leakage. Provenance/metadata ignored."""
    issues: list[str] = []
    eng = extract_engineering_capability(record)
    args = _collect_args(record)
    caps = [c.strip() for c in (eng or "").split(",") if c.strip()] or [None]
    for cap in caps:
        try:
            assert_no_forbidden_how(capability_id=cap, args=args)
        except ForbiddenHowError as exc:
            issues.append(f"forbidden_how:{exc}")
    text = _model_facing_text(record)
    for token in sorted(_TEXT_HOW_TOKENS):
        if token in text:
            # Ignore if token only appears as part of a preferred capability id
            # already validated (e.g. none of these appear in preferred IDs).
            issues.append(f"forbidden_how_token:{token}")
    # Explicit backend action as capability (not preferred)
    for cap in caps:
        if not cap:
            continue
        if cap in FORBIDDEN_BACKEND_ACTIONS and cap not in PREFERRED_ENGINEERING_CAPABILITY_IDS:
            issues.append(f"forbidden_backend_as_capability:{cap}")
        if any(k in FORBIDDEN_ARG_KEYS for k in args):
            # already covered by assert_no_forbidden_how; keep single form
            pass
    return sorted(set(issues))


def scan_taxonomy(record: Mapping[str, Any]) -> list[str]:
    """Dedicated provider vs Engineering separation checks."""
    issues: list[str] = []
    provider = str(record.get("provider_capability") or "").strip()
    if provider and provider not in PROVIDER_CAPABILITY_IDS:
        issues.append(f"invalid_provider_capability:{provider}")
    eng = extract_engineering_capability(record)
    rtype = str(record.get("record_type") or "")
    if rtype in {"capability_intent", "execution_work_unit"} and eng:
        for part in eng.split(","):
            part = part.strip()
            if part and part not in PREFERRED_ENGINEERING_CAPABILITY_IDS:
                issues.append(f"non_preferred_engineering_capability:{part}")
            if part in PROVIDER_CAPABILITY_IDS:
                issues.append(f"provider_used_as_engineering:{part}")
    if rtype == "planning_decision":
        expected = record.get("expected_output") if isinstance(record.get("expected_output"), Mapping) else {}
        for idx, step in enumerate(expected.get("steps") or []):
            if not isinstance(step, Mapping):
                continue
            act = str(step.get("action") or step.get("capability_id") or "").strip()
            if act and act not in PREFERRED_ENGINEERING_CAPABILITY_IDS:
                issues.append(f"planning_step_non_preferred:{idx}:{act}")
            if act in PROVIDER_CAPABILITY_IDS:
                issues.append(f"planning_step_provider_pack:{idx}:{act}")
    if provider and classify_capability_id(provider) is TaxonomyPlane.ENGINEERING:
        issues.append(f"engineering_used_as_provider:{provider}")
    return sorted(set(issues))


def provenance_ok_for_projected(record: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    prov = record.get("provenance")
    if not isinstance(prov, Mapping):
        return ["missing_provenance"]
    for key in (
        "source_dataset",
        "source_record_id",
        "source_schema_version",
        "projection_version",
        "projection_status",
    ):
        if not str(prov.get(key) or "").strip():
            issues.append(f"missing_provenance_field:{key}")
    return issues
