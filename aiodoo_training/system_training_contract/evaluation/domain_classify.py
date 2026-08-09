"""TR-7 domain classification — GENERIC / ODOO_SPECIALIZED / AMBIGUOUS.

TR-6 used overly narrow cues (literal ``odoo`` / ``manifest`` / ``addons/`` only),
which falsely flagged legitimate Odoo examples (e.g. ``models/partner.py``,
``action_confirm``) as ambiguous. TR-7 classifies from semantic content.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

__all__ = [
    "DomainClass",
    "ODOO_SEMANTIC_CUES",
    "classify_domain",
    "semantic_blob",
    "has_odoo_semantics",
]


class DomainClass(StrEnum):
    GENERIC = "GENERIC"
    ODOO_SPECIALIZED = "ODOO_SPECIALIZED"
    AMBIGUOUS = "AMBIGUOUS"


# Content cues that justify Odoo specialization (not provenance alone).
ODOO_SEMANTIC_CUES: tuple[str, ...] = (
    "odoo",
    "__manifest__",
    "addons/",
    "models/partner",
    "models/sale",
    "sale_order",
    "action_confirm",
    "res.partner",
    "ir.model",
    "api.one",
    "api.multi",
    "api.depends",
    "api.model",
    "sudo(",
    "env['",
    'env["',
    "self.env",
    "@api.",
    "odoo.addons",
    "module manifest",
    "odoo module",
    "odoo worker",
)


def semantic_blob(record: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "input": record.get("input"),
            "expected_output": record.get("expected_output"),
            "evidence": record.get("evidence"),
            "objective": (record.get("input") or {}).get("objective")
            if isinstance(record.get("input"), Mapping)
            else None,
        },
        sort_keys=True,
        default=str,
    ).lower()


def has_odoo_semantics(record: Mapping[str, Any]) -> bool:
    blob = semantic_blob(record)
    return any(cue in blob for cue in ODOO_SEMANTIC_CUES)


def classify_domain(record: Mapping[str, Any]) -> DomainClass:
    """Classify from semantic content + existing domain_specialization label."""
    labeled_odoo = record.get("domain_specialization") == "odoo"
    cues = has_odoo_semantics(record)
    if labeled_odoo and cues:
        return DomainClass.ODOO_SPECIALIZED
    if not labeled_odoo and not cues:
        return DomainClass.GENERIC
    # Label says odoo but no cues, or cues without label → ambiguous until corrected
    return DomainClass.AMBIGUOUS


def corrected_domain_specialization(record: Mapping[str, Any]) -> tuple[str | None, DomainClass, str]:
    """Return (domain_specialization value, class, action).

    Actions: keep | set_odoo | clear_odoo | quarantine
    """
    labeled_odoo = record.get("domain_specialization") == "odoo"
    cues = has_odoo_semantics(record)
    if cues:
        # Provable Odoo specialization
        if labeled_odoo:
            return "odoo", DomainClass.ODOO_SPECIALIZED, "keep"
        return "odoo", DomainClass.ODOO_SPECIALIZED, "set_odoo"
    if labeled_odoo and not cues:
        # Label without semantic justification — clear to generic (not fabricate Odoo)
        return None, DomainClass.GENERIC, "clear_odoo"
    return None, DomainClass.GENERIC, "keep"
