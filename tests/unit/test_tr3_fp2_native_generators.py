"""TR-3 — FP2-native generators + fixture corpus tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiodoo_training.system_training_contract import (
    SYSTEM_TRAINING_CONTRACT_VERSION,
    ProjectionStatus,
    TaxonomyPlane,
    classify_capability_id,
    validate_record_mapping,
)
from aiodoo_training.system_training_contract.forbidden import ForbiddenHowError, assert_no_forbidden_how
from aiodoo_training.system_training_contract.generators import (
    DATASET_GENERATION_VERSION,
    DEVELOPMENT_RECORD_TYPES,
    GENERATOR_NAMES,
    REASONING_RECORD_TYPES,
    emit_all_fixtures,
    generate_all,
    record_provider_capabilities,
)
from aiodoo_training.system_training_contract.generators.emit import generate_projection_fixtures
from aiodoo_training.system_training_contract.generators.mapping import assert_no_adapter_chain
from aiodoo_training.system_training_contract.taxonomy import (
    DEVELOPMENT_PROVIDER_CAPABILITIES,
    PREFERRED_ENGINEERING_CAPABILITY_IDS,
    REASONING_PROVIDER_CAPABILITIES,
)


def test_generate_all_families_valid_and_deterministic() -> None:
    first = generate_all()
    second = generate_all()
    assert set(first) == set(GENERATOR_NAMES)
    for name in GENERATOR_NAMES:
        assert len(first[name]) >= 10
        assert len(first[name]) <= 40
        assert [json.dumps(r, sort_keys=True) for r in first[name]] == [
            json.dumps(r, sort_keys=True) for r in second[name]
        ]
        for rec in first[name]:
            assert rec["training_contract_version"] == SYSTEM_TRAINING_CONTRACT_VERSION
            validate_record_mapping(rec)


def test_capability_intents_use_preferred_engineering_only() -> None:
    for rec in generate_all()["capability_intent"]:
        cid = rec["expected_output"]["capability_id"]
        assert cid in PREFERRED_ENGINEERING_CAPABILITY_IDS
        assert classify_capability_id(cid) is TaxonomyPlane.ENGINEERING
        assert classify_capability_id(rec["provider_capability"]) is TaxonomyPlane.PROVIDER


def test_work_unit_and_planning_structure() -> None:
    families = generate_all()
    for rec in families["execution_work_unit"]:
        assert rec["record_type"] == "execution_work_unit"
        assert rec["expected_output"]["capability_id"] in PREFERRED_ENGINEERING_CAPABILITY_IDS
        assert "objective" in rec["input"]
    complete = [
        r
        for r in families["planning_decision"]
        if r["expected_output"]["decision_kind"] == "complete"
    ]
    assert complete
    assert complete[0]["expected_output"]["steps"] == []


def test_observation_feedback_state_decision_loop() -> None:
    families = generate_all()
    assert any(r["evidence"]["status"] == "failed" for r in families["observation"])
    assert any(
        r["evidence"]["objective_state"] == "incomplete"
        and r["evidence"]["execution_state"] == "succeeded"
        for r in families["engineering_feedback"]
    )
    states = {r["metadata"]["scenario"]: r for r in families["engineering_state"]}
    assert states["cycle1_validation_failed"]["evidence"]["cycle_index"] == 1
    assert (
        states["cycle3_validation_passed"]["evidence"]["current_fields"]["validation_status"]
        == "passed"
    )
    assert states["previous_complete_current_failure"]["evidence"]["objective_state"] == "failed"
    assert (
        states["previous_failure_current_success"]["evidence"]["objective_state"] == "complete"
    )
    for rec in families["decision_context"]:
        blob = json.dumps(rec)
        for token in ("local_workspace", "implementation_id", "stdout", "password"):
            assert token not in blob
    kinds = {r["expected_output"]["decision_kind"] for r in families["loop_decision"]}
    assert {"replan", "complete", "escalate", "retry", "recover"} <= kinds


def test_provider_engineering_and_adapter_mapping() -> None:
    assert_no_adapter_chain()
    assert DEVELOPMENT_PROVIDER_CAPABILITIES.isdisjoint(REASONING_PROVIDER_CAPABILITIES)
    assert "planning_decision" in REASONING_RECORD_TYPES
    assert "execution_work_unit" in DEVELOPMENT_RECORD_TYPES
    assert "planner" in record_provider_capabilities("planning_decision")
    assert "execution" in record_provider_capabilities("execution_work_unit")


def test_forbidden_how_still_rejected() -> None:
    with pytest.raises(ForbiddenHowError):
        assert_no_forbidden_how(capability_id="local_git")


def test_projection_fixtures_cover_all_statuses() -> None:
    fixtures = generate_projection_fixtures()
    statuses = {f["status"] for f in fixtures}
    assert statuses == {
        ProjectionStatus.PROJECTED.value,
        ProjectionStatus.PARTIALLY_PROJECTED.value,
        ProjectionStatus.UNSUPPORTED.value,
        ProjectionStatus.REJECTED.value,
    }
    for f in fixtures:
        assert f["provenance"]["source_record_id"]
        assert f["provenance"]["source_dataset"]
        assert f["provenance"]["projection_version"]


def test_emit_fixtures_to_tmp_and_preserve_contract(tmp_path: Path) -> None:
    counts = emit_all_fixtures(tmp_path)
    assert counts["capability_intent"] >= 10
    assert (tmp_path / "manifest.json").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_contract_version"] == SYSTEM_TRAINING_CONTRACT_VERSION
    assert manifest["dataset_generation_version"] == DATASET_GENERATION_VERSION
    assert manifest["legacy_datasets_untouched"] is True
    line = (tmp_path / "capability_intent.jsonl").read_text(encoding="utf-8").splitlines()[0]
    validate_record_mapping(json.loads(line))


def test_odoo_specialization_present_but_not_required() -> None:
    families = generate_all()
    odoo = [
        r
        for r in families["capability_intent"]
        if r.get("domain_specialization") == "odoo"
    ]
    generic = [
        r for r in families["capability_intent"] if not r.get("domain_specialization")
    ]
    assert odoo and generic
