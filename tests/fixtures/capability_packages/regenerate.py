#!/usr/bin/env python3
"""Regenerate representative protocol/v1 Capability Package goldens from builders.

Run from the aiodoo-training repository root:

    python3 tests/fixtures/capability_packages/regenerate.py

Does not modify production code. Goldens omit volatile producer fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from aiodoo_training.artifacts.publish_contract import (  # noqa: E402
    build_adapter_artifact_json,
    build_base_model_artifact_json,
    build_merged_artifact_json,
)

PROTOCOL_V1 = Path(__file__).resolve().parent / "protocol" / "v1"
CREATED = "2026-07-19T12:00:00Z"
_STRIP = frozenset({"training_version", "producer", "source_checkpoint", "source_bundle"})


def _resolved(capability: str) -> dict:
    return {
        "experiment": {"id": capability},
        "datasets": [{"dataset_type": capability}],
        "model": {"family": "qwen", "identifier": "Qwen/Qwen3-8B"},
        "adaptation": {"adapter_type": "qlora", "strategy": "qlora"},
        "dataset_version": "v1.0.0",
    }


def _write(rel: str, payload: dict) -> None:
    path = PROTOCOL_V1 / rel / "artifact.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in payload.items() if k not in _STRIP}
    path.write_text(json.dumps(clean, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    _write(
        "base_models/qwen3-8b",
        build_base_model_artifact_json(
            model_id="Qwen/Qwen3-8B",
            model_family="qwen",
            architecture="qwen",
            created_at=CREATED,
        ),
    )
    _write(
        "adapters/coding",
        build_adapter_artifact_json(
            experiment_id="aiodoo-coding",
            resolved=_resolved("coding"),
            created_at=CREATED,
        ),
    )
    _write(
        "adapters/repair",
        build_adapter_artifact_json(
            experiment_id="aiodoo-repair",
            resolved=_resolved("repair"),
            created_at=CREATED,
        ),
    )
    _write(
        "merged/coding",
        build_merged_artifact_json(
            experiment_id="aiodoo-coding",
            resolved=_resolved("coding"),
            created_at=CREATED,
        ),
    )


if __name__ == "__main__":
    main()
