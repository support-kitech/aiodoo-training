"""Shared helpers for FP2-native generators."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

DATASET_GENERATION_VERSION: str = "fp2-native-1.0.0"


def fixture_metadata(
    *,
    generator: str,
    index: int,
    provider_capability: str | None,
    domain_specialization: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "dataset_generation_version": DATASET_GENERATION_VERSION,
        "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
        "generator": generator,
        "fixture_index": index,
        "source": "fp2_native_fixture",
        "legacy": False,
    }
    if provider_capability:
        meta["provider_capability"] = provider_capability
    if domain_specialization:
        meta["domain_specialization"] = domain_specialization
    if extra:
        meta.update(dict(extra))
    return meta


def rid(prefix: str, index: int) -> str:
    return f"fp2-{prefix}-{index:03d}"


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(r), sort_keys=True, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)
