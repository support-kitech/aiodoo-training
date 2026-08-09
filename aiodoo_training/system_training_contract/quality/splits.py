"""Deterministic train/validation/test split strategy (design + assignment)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class SplitAssignment(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def document_split_strategy() -> dict[str, Any]:
    """Documented strategy — no large corpus generation in TR-4."""
    return {
        "version": "fp2-split-1.0.0",
        "method": "scenario_hash_bucket",
        "ratios": [
            "Hash stable scenario key (record_type + normalized objective/goal + "
            "capability fingerprint + domain_specialization).",
            "Bucket: train 80% / validation 10% / test 10% by first nibble of SHA-256.",
            "All records sharing the same scenario key get the same split "
            "(prevents multi-cycle leakage across splits).",
            "Negative/adversarial quality fixtures are NEVER assigned to train "
            "(held in quality_only bucket).",
            "Projection fixtures are quality/audit only unless status=projected "
            "and explicitly promoted later.",
        ],
        "ratios": {"train": 0.80, "validation": 0.10, "test": 0.10},
        "leakage_prevention": (
            "Multi-cycle scenarios (cycle1/2/3) share scenario family keys so they "
            "cannot straddle train and test."
        ),
    }


def scenario_key(record: Mapping[str, Any]) -> str:
    meta = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    family = str(meta.get("scenario_family") or "").strip()
    if family:
        return f"family:{family}"
    scenario = str(meta.get("scenario") or "")
    if scenario.startswith("cycle") or scenario.startswith("previous_"):
        # Group multi-cycle narratives
        family = "partner_field_validate" if "cycle" in scenario or "previous_" in scenario else scenario
        if "partner" in str(record).lower() or "validation" in scenario or "repair" in scenario:
            family = "narrative_partner_validate"
        if scenario.startswith("previous_"):
            family = f"narrative_{scenario.split('_')[1]}"  # complete / failure
        return f"family:{family}"
    rtype = str(record.get("record_type") or "")
    inp = record.get("input") if isinstance(record.get("input"), Mapping) else {}
    evidence = record.get("evidence") if isinstance(record.get("evidence"), Mapping) else {}
    expected = record.get("expected_output") if isinstance(record.get("expected_output"), Mapping) else {}
    objective = str(
        inp.get("objective") or inp.get("goal") or evidence.get("objective") or expected.get("decision_kind") or ""
    ).strip().lower()
    cap = str(expected.get("capability_id") or evidence.get("capability_id") or "")
    domain = str(record.get("domain_specialization") or "generic")
    return f"{rtype}|{cap}|{objective}|{domain}"


def assign_split(record: Mapping[str, Any]) -> SplitAssignment:
    meta = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    if str(meta.get("quality_corpus") or "").startswith("negative"):
        # Not a training split — callers should exclude; map to test for safety
        return SplitAssignment.TEST
    key = scenario_key(record)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # First byte → 0-255; 0-203 train (~80%), 204-229 val (~10%), 230-255 test (~10%)
    bucket = int(digest[:2], 16)
    if bucket <= 203:
        return SplitAssignment.TRAIN
    if bucket <= 229:
        return SplitAssignment.VALIDATION
    return SplitAssignment.TEST
