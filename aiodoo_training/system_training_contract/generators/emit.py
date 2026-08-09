"""Emit all FP2-native fixture corpora + selective projection fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from aiodoo_training.system_training_contract.generators.capability_intent import (
    generate_capability_intents,
)
from aiodoo_training.system_training_contract.generators.common import (
    DATASET_GENERATION_VERSION,
    write_jsonl,
)
from aiodoo_training.system_training_contract.generators.decision_context import (
    generate_decision_contexts,
)
from aiodoo_training.system_training_contract.generators.feedback import generate_feedback
from aiodoo_training.system_training_contract.generators.loop_decision import (
    generate_loop_decisions,
)
from aiodoo_training.system_training_contract.generators.observation import (
    generate_observations,
)
from aiodoo_training.system_training_contract.generators.planning import (
    generate_planning_decisions,
)
from aiodoo_training.system_training_contract.generators.state import generate_states
from aiodoo_training.system_training_contract.generators.work_unit import generate_work_units
from aiodoo_training.system_training_contract.projection import (
    ProjectionStatus,
    project_historical_record,
)
from aiodoo_training.system_training_contract.records import validate_record_mapping
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

GENERATOR_NAMES: tuple[str, ...] = (
    "capability_intent",
    "execution_work_unit",
    "planning_decision",
    "observation",
    "engineering_feedback",
    "engineering_state",
    "decision_context",
    "loop_decision",
)

_GENERATORS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "capability_intent": generate_capability_intents,
    "execution_work_unit": generate_work_units,
    "planning_decision": generate_planning_decisions,
    "observation": generate_observations,
    "engineering_feedback": generate_feedback,
    "engineering_state": generate_states,
    "decision_context": generate_decision_contexts,
    "loop_decision": generate_loop_decisions,
}

# Selective historical projection fixtures (do not touch production JSONL).
_HISTORICAL_CASES: tuple[tuple[str, str, dict[str, Any], ProjectionStatus], ...] = (
    (
        "hist_eng_ok",
        "misc_fp2_tagged",
        {
            "capability_id": "workspace.write",
            "objective": "Write file",
            "args": {"path": "a.py"},
        },
        ProjectionStatus.PROJECTED,
    ),
    (
        "hist_coding_partial",
        "coding_sft",
        {"objective": "implement helper", "code": "def f(): ..."},
        ProjectionStatus.PARTIALLY_PROJECTED,
    ),
    (
        "hist_planner_unsupported",
        "planner_odoo_v1",
        {"steps": [{"action": "create_file", "path": "x.py"}]},
        ProjectionStatus.UNSUPPORTED,
    ),
    (
        "hist_execution_unsupported",
        "execution_train",
        {"op": "apply_artifact", "artifact_id": "a1"},
        ProjectionStatus.UNSUPPORTED,
    ),
    (
        "hist_forbidden_rejected",
        "misc",
        {"capability_id": "local_workspace"},
        ProjectionStatus.REJECTED,
    ),
    (
        "hist_approval_projected",
        "approval_sft",
        {"decision_kind": "approve", "reason": "looks good"},
        ProjectionStatus.PROJECTED,
    ),
    (
        "hist_repair_explicit",
        "repair_sft",
        {
            "capability_id": "execution.repair",
            "objective": "fix import",
            "args": {"path": "m.py"},
        },
        ProjectionStatus.PROJECTED,
    ),
)


def generate_all() -> dict[str, list[dict[str, Any]]]:
    """Generate and validate all FP2-native fixture families."""
    result: dict[str, list[dict[str, Any]]] = {}
    for name in GENERATOR_NAMES:
        records = _GENERATORS[name]()
        validated = [validate_record_mapping(r) for r in records]
        # Determinism: regenerating must match first pass serialization.
        again = [validate_record_mapping(r) for r in _GENERATORS[name]()]
        if [json.dumps(r, sort_keys=True) for r in validated] != [
            json.dumps(r, sort_keys=True) for r in again
        ]:
            raise RuntimeError(f"nondeterministic generator: {name}")
        result[name] = validated
    return result


def generate_projection_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case_id, dataset, record, expected in _HISTORICAL_CASES:
        result = project_historical_record(
            record,
            source_dataset=dataset,
            source_record_id=case_id,
            source_schema_version="legacy",
        )
        if result.status is not expected:
            raise RuntimeError(
                f"projection fixture {case_id}: expected {expected}, got {result.status}"
            )
        payload = result.to_dict()
        payload["fixture_id"] = case_id
        payload["expected_status"] = expected.value
        out.append(payload)
    return out


def emit_all_fixtures(output_dir: str | Path) -> Mapping[str, int]:
    """Write FP2-native corpora under ``output_dir`` (typically datasets/fp2)."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    families = generate_all()
    for name, records in families.items():
        path = root / f"{name}.jsonl"
        counts[name] = write_jsonl(path, records)

    proj = generate_projection_fixtures()
    counts["projection"] = write_jsonl(root / "projection_fixtures.jsonl", proj)

    manifest = {
        "dataset_generation_version": DATASET_GENERATION_VERSION,
        "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
        "classification": "fp2_native_fixture",
        "legacy_datasets_untouched": True,
        "record_counts": counts,
        "generators": list(GENERATOR_NAMES),
        "notes": (
            "Experimental FP2-native fixtures. Not production Certified V1 corpora. "
            "Do not mix silently with legacy provider JSONL."
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts["manifest"] = 1
    return counts
