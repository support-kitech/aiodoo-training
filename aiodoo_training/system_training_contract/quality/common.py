"""Shared helpers for quality evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

NATIVE_FAMILIES: tuple[str, ...] = (
    "capability_intent",
    "execution_work_unit",
    "planning_decision",
    "observation",
    "engineering_feedback",
    "engineering_state",
    "decision_context",
    "loop_decision",
)


class GateOutcome(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def stable_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def fingerprint_record(record: Mapping[str, Any]) -> str:
    """Semantic fingerprint — ignores record_id / provenance / metadata noise."""
    rtype = str(record.get("record_type") or "")
    payload: dict[str, Any] = {
        "record_type": rtype,
        "provider_capability": record.get("provider_capability"),
        "domain_specialization": record.get("domain_specialization"),
        "system_contract": record.get("system_contract"),
        "input": record.get("input"),
        "expected_output": record.get("expected_output"),
        "evidence": _strip_evidence_noise(record.get("evidence")),
    }
    expected = record.get("expected_output") if isinstance(record.get("expected_output"), Mapping) else {}
    evidence = record.get("evidence") if isinstance(record.get("evidence"), Mapping) else {}
    inp = record.get("input") if isinstance(record.get("input"), Mapping) else {}
    capability = expected.get("capability_id") or evidence.get("capability_id")
    if not capability:
        steps = expected.get("steps")
        if isinstance(steps, list) and steps and isinstance(steps[0], Mapping):
            capability = steps[0].get("action") or steps[0].get("capability_id")
    payload["capability"] = capability or ""
    payload["objective"] = (
        inp.get("objective")
        or inp.get("goal")
        or evidence.get("objective")
        or expected.get("decision_kind")
        or ""
    )
    digest = hashlib.sha256(stable_dumps(payload).encode("utf-8")).hexdigest()
    return digest[:24]


def _strip_evidence_noise(evidence: Any) -> Any:
    if not isinstance(evidence, Mapping):
        return evidence
    out = dict(evidence)
    # Keep semantic fields; drop nothing critical for duplicate detection.
    return out


def extract_engineering_capability(record: Mapping[str, Any]) -> str | None:
    expected = record.get("expected_output") if isinstance(record.get("expected_output"), Mapping) else {}
    evidence = record.get("evidence") if isinstance(record.get("evidence"), Mapping) else {}
    for key in ("capability_id",):
        val = expected.get(key) or evidence.get(key)
        if val:
            return str(val)
    steps = expected.get("steps")
    if isinstance(steps, list):
        caps: list[str] = []
        for step in steps:
            if isinstance(step, Mapping):
                act = step.get("action") or step.get("capability_id")
                if act:
                    caps.append(str(act))
        if caps:
            return ",".join(caps)
    return None


def model_facing_blob(record: Mapping[str, Any]) -> str:
    """Concatenate model-facing fields for HOW scanning (exclude provenance/metadata)."""
    parts = [
        record.get("input"),
        record.get("expected_output"),
        record.get("evidence"),
        record.get("system_contract"),
        record.get("provider_capability"),
    ]
    # Include capability_id in expected/evidence already
    return stable_dumps(parts).lower()
