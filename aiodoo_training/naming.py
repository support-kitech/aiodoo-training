"""Canonical public training identifiers and Drive artifact naming.

Public surface uses semantic training IDs (``coding``, ``planner``, …).
Numeric ``EXP-NNNN`` values are internal bookkeeping only and must never appear
in Drive paths, adapter product names, notebook UI, or public CLI strings.
"""

from __future__ import annotations

from typing import Final

# Permanent public training identifiers (configuration + cache keys).
TRAINING_IDS: Final[tuple[str, ...]] = (
    "coding",
    "planner",
    "context",
    "conversation",
    "repair",
    "execution",
    "approval",
    "evaluation",
)

_TRAINING_ID_SET: Final[frozenset[str]] = frozenset(TRAINING_IDS)

# Adapter products are published under models/adapters/{ADAPTER_PRODUCT_PREFIX}{id}/
ADAPTER_PRODUCT_PREFIX: Final[str] = "aiodoo-"

# Migration map: legacy public EXP ids → semantic training ids.
# Kept for resume / config path resolution; never exposed as Drive folder names.
LEGACY_INTERNAL_ID_TO_TRAINING_ID: Final[dict[str, str]] = {
    "EXP-0001": "coding",
}

TRAINING_ID_TO_INTERNAL_ID: Final[dict[str, str]] = {
    training_id: internal_id
    for internal_id, training_id in LEGACY_INTERNAL_ID_TO_TRAINING_ID.items()
}

# Relative config root under the aiodoo-training repository.
TRAINING_CONFIG_ROOT: Final[str] = "configs/training"

# Legacy config root (read-only migration lookup).
LEGACY_EXPERIMENT_CONFIG_ROOT: Final[str] = "configs/experiments/production"


def is_training_id(value: str) -> bool:
    """Return True when ``value`` is a canonical public training id."""
    return value in _TRAINING_ID_SET


def is_legacy_internal_id(value: str) -> bool:
    """Return True when ``value`` looks like a legacy EXP-NNNN internal id."""
    return value.startswith("EXP-") and value in LEGACY_INTERNAL_ID_TO_TRAINING_ID


def normalize_training_id(value: str) -> str:
    """
    Resolve a user or config identifier to a canonical training id.

    Accepts semantic ids (``coding``) and legacy internal ids (``EXP-0001``).
    """
    raw = value.strip()
    if not raw:
        raise ValueError("training id must be a non-empty string")
    if is_training_id(raw):
        return raw
    if raw in LEGACY_INTERNAL_ID_TO_TRAINING_ID:
        return LEGACY_INTERNAL_ID_TO_TRAINING_ID[raw]
    # Allow future EXP-NNNN → unknown mapping to fail clearly.
    if raw.startswith("EXP-"):
        raise ValueError(
            f"Unknown legacy internal id {raw!r}; known: {sorted(LEGACY_INTERNAL_ID_TO_TRAINING_ID)}"
        )
    raise ValueError(
        f"Unknown training id {raw!r}; expected one of {list(TRAINING_IDS)}"
    )


def adapter_product_id(training_id: str) -> str:
    """Public adapter product directory name, e.g. ``coding`` → ``aiodoo-coding``."""
    tid = normalize_training_id(training_id)
    return f"{ADAPTER_PRODUCT_PREFIX}{tid}"


def internal_id_for(training_id: str) -> str | None:
    """Return legacy internal EXP id for bookkeeping, if one exists."""
    tid = normalize_training_id(training_id)
    return TRAINING_ID_TO_INTERNAL_ID.get(tid)


def stage_display_name(training_id: str) -> str:
    """Human-readable stage label for notebook UI (``coding`` → ``Coding``)."""
    tid = normalize_training_id(training_id)
    return tid.replace("_", " ").title()


def resolve_public_training_id(resolved: dict) -> str:
    """
    Extract the public training id from a resolved experiment config mapping.

    Preference:
    1. ``experiment.id`` / ``experiment.training_id``
    2. ``metadata.internal_id`` / ``experiment.internal_id`` (legacy EXP map)
    3. ``name`` when semantic or legacy
    4. ``experiment.stage`` when it is a known training id (not ``production``)

    Note: ``training`` in configs is the hyperparameter block — do not read
    ``training.id`` here (that field is not part of the identity contract).
    """
    experiment = resolved.get("experiment")
    if isinstance(experiment, dict):
        for key in ("id", "training_id"):
            value = experiment.get(key)
            if isinstance(value, str) and value.strip():
                try:
                    return normalize_training_id(value)
                except ValueError:
                    continue
        for key in ("internal_id",):
            value = experiment.get(key)
            if isinstance(value, str) and value.strip():
                try:
                    return normalize_training_id(value)
                except ValueError:
                    continue
        stage = experiment.get("stage")
        if isinstance(stage, str) and stage.strip() and is_training_id(stage.strip()):
            return stage.strip()

    metadata = resolved.get("metadata")
    if isinstance(metadata, dict):
        internal = metadata.get("internal_id")
        if isinstance(internal, str) and internal.strip():
            try:
                return normalize_training_id(internal)
            except ValueError:
                pass

    name = resolved.get("name")
    if isinstance(name, str) and name.strip():
        return normalize_training_id(name)

    raise ValueError(
        "Resolved config is missing a public training id "
        "(expected experiment.id)."
    )


__all__ = [
    "ADAPTER_PRODUCT_PREFIX",
    "LEGACY_EXPERIMENT_CONFIG_ROOT",
    "LEGACY_INTERNAL_ID_TO_TRAINING_ID",
    "TRAINING_CONFIG_ROOT",
    "TRAINING_IDS",
    "TRAINING_ID_TO_INTERNAL_ID",
    "adapter_product_id",
    "internal_id_for",
    "is_legacy_internal_id",
    "is_training_id",
    "normalize_training_id",
    "resolve_public_training_id",
    "stage_display_name",
]
