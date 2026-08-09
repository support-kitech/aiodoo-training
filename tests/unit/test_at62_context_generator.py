"""AT-6.2 — FP2 Context provider mapping, generator, fixtures, legacy separation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from aiodoo_training.system_training_contract import (
    SYSTEM_TRAINING_CONTRACT_VERSION,
    TaxonomyPlane,
    classify_capability_id,
    validate_record_mapping,
)
from aiodoo_training.system_training_contract.forbidden import assert_no_forbidden_how
from aiodoo_training.system_training_contract.generators.context import (
    CONTEXT_CORPUS_VERSION,
    emit_context_fixtures,
    generate_context_records,
)
from aiodoo_training.system_training_contract.generators.mapping import (
    CONTEXT_ALLOWED_RECORD_TYPES,
    CONTEXT_REJECTED_RECORD_TYPES,
    assert_no_adapter_chain,
    record_provider_capabilities,
)
from aiodoo_training.system_training_contract.quality.formatters import format_fp2_pack
from aiodoo_training.system_training_contract.quality.gates import (
    scan_forbidden_how,
    scan_taxonomy,
)
from aiodoo_training.system_training_contract.quality.negatives import (
    NEGATIVE_CASES,
    evaluate_negative_case,
)
from aiodoo_training.system_training_contract.taxonomy import (
    DEVELOPMENT_PROVIDER_CAPABILITIES,
    PREFERRED_ENGINEERING_CAPABILITY_IDS,
)

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]
BATCH2 = WORKSPACE / "aiodoo-datasets" / "datasets" / "fp2" / "controlled_batch_2"
LEGACY_CONTEXT = WORKSPACE / "aiodoo-datasets" / "datasets" / "context_v1_0.jsonl"
BATCH2_CHECKSUM = "728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227"


def _batch2_corpus_checksum() -> str:
    manifest = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest.get("checksum") or "")


def test_context_in_development_taxonomy() -> None:
    assert "context" in DEVELOPMENT_PROVIDER_CAPABILITIES
    assert classify_capability_id("context") is TaxonomyPlane.PROVIDER


def test_context_mapping_allow_and_reject() -> None:
    assert_no_adapter_chain()
    for rtype in CONTEXT_ALLOWED_RECORD_TYPES:
        assert "context" in record_provider_capabilities(rtype)
    for rtype in CONTEXT_REJECTED_RECORD_TYPES:
        assert "context" not in record_provider_capabilities(rtype)
    # Homonym: decision_context record_type ≠ provider context
    assert "context" not in record_provider_capabilities("decision_context")
    # Contract: Context is not Work Units
    assert "context" not in record_provider_capabilities("execution_work_unit")


def test_generate_context_records_valid_and_deterministic() -> None:
    first = generate_context_records()
    second = generate_context_records()
    assert 20 <= len(first) <= 40
    assert [json.dumps(r, sort_keys=True) for r in first] == [
        json.dumps(r, sort_keys=True) for r in second
    ]
    ids = [r["record_id"] for r in first]
    assert len(ids) == len(set(ids))
    by_type = Counter(r["record_type"] for r in first)
    assert set(by_type) <= set(CONTEXT_ALLOWED_RECORD_TYPES)
    assert by_type["capability_intent"] >= 10
    assert by_type["observation"] >= 8
    for rec in first:
        assert rec["training_contract_version"] == SYSTEM_TRAINING_CONTRACT_VERSION
        assert rec["provider_capability"] == "context"
        assert rec["record_type"] in CONTEXT_ALLOWED_RECORD_TYPES
        assert rec["metadata"].get("legacy") is False
        validate_record_mapping(rec)
        assert not scan_forbidden_how(rec)
        assert not scan_taxonomy(rec)
        if rec["record_type"] == "capability_intent":
            cid = rec["expected_output"]["capability_id"]
            assert cid in PREFERRED_ENGINEERING_CAPABILITY_IDS
            assert classify_capability_id(cid) is TaxonomyPlane.ENGINEERING


def test_context_odoo_and_generic_present() -> None:
    records = generate_context_records()
    odoo = [r for r in records if r.get("domain_specialization") == "odoo"]
    generic = [r for r in records if not r.get("domain_specialization")]
    assert odoo and generic


def test_context_engineering_capabilities_are_locate_family() -> None:
    allowed = {
        "workspace.search",
        "workspace.navigate",
        "workspace.read",
        "repository.inspect",
    }
    records = generate_context_records()
    for rec in records:
        if rec["record_type"] == "capability_intent":
            assert rec["expected_output"]["capability_id"] in allowed
        elif rec["record_type"] == "observation" and rec["evidence"].get("capability_id"):
            assert rec["evidence"]["capability_id"] in allowed


def test_decision_context_provider_context_policy_rejected() -> None:
    """Schema may accept provider=context; mapping must not allow the family."""
    assert "context" not in record_provider_capabilities("decision_context")
    bad = {
        "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
        "record_type": "decision_context",
        "record_id": "policy-ctx-dc",
        "system_contract": "execution.engineering_decision_context",
        "provider_capability": "context",
        "input": {
            "objective": "x",
            "objective_state": "incomplete",
            "cycle_index": 1,
            "execution_state": "failed",
            "observation_quality": "failed",
            "validation_status": "failed",
            "repair_status": "not_started",
            "blockers": [],
            "failures": ["validation_failed"],
            "missing_outcomes": ["validation"],
            "expected_outcomes": {},
            "possible_next_actions": ["replan"],
            "continuation_hint": "replan",
            "bounded_history": [],
        },
        "expected_output": {},
        "provenance": {},
        "metadata": {},
    }
    # Schema validates provider id, but family is rejected by mapping policy.
    validate_record_mapping(bad)
    assert bad["record_type"] in CONTEXT_REJECTED_RECORD_TYPES


def test_negative_context_forbidden_how_rejected() -> None:
    case = next(c for c in NEGATIVE_CASES if c["case_id"] == "neg_context_forbidden_how")
    result = evaluate_negative_case(case)
    assert result["matched"] is True
    assert result["accepted"] is False


def test_negative_control_positive_context_accepted() -> None:
    case = next(c for c in NEGATIVE_CASES if c["case_id"] == "pos_context_locate_intent")
    result = evaluate_negative_case(case)
    assert result["matched"] is True
    assert result["accepted"] is True


def test_emit_context_fixtures_tmp_and_not_batch2(tmp_path: Path) -> None:
    train = tmp_path / "train_ctx"
    data = tmp_path / "data_ctx"
    before = _batch2_corpus_checksum()
    result = emit_context_fixtures(training_fixtures_root=train, datasets_root=data)
    assert result["version"] == CONTEXT_CORPUS_VERSION
    assert result["count"] == len(generate_context_records())
    assert (train / "context_native.jsonl").is_file()
    assert (data / "manifest.json").is_file()
    man = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert man["legacy_projection"] is False
    assert man["controlled_batch_2_modified"] is False
    assert _batch2_corpus_checksum() == before == BATCH2_CHECKSUM
    # Emit must never write into controlled_batch_2
    assert not any("controlled_batch_2" in p for p in result["written"])


def test_legacy_context_untouched_and_not_fp2() -> None:
    assert LEGACY_CONTEXT.is_file()
    line = LEGACY_CONTEXT.read_text(encoding="utf-8").splitlines()[0]
    legacy = json.loads(line)
    assert "query" in legacy and "artifacts" in legacy
    assert "provider_capability" not in legacy
    assert "record_type" not in legacy
    assert "training_contract_version" not in legacy
    for rec in generate_context_records():
        assert rec.get("metadata", {}).get("legacy") is False
        blob = json.dumps(rec)
        assert "context_v1_0" not in blob


def test_controlled_batch_2_immutable_checksum() -> None:
    assert _batch2_corpus_checksum() == BATCH2_CHECKSUM
    # No context labels in TR-7 evidence corpus
    native_files = [
        "capability_intent.jsonl",
        "execution_work_unit.jsonl",
        "observation.jsonl",
        "planning_decision.jsonl",
        "engineering_feedback.jsonl",
        "engineering_state.jsonl",
        "decision_context.jsonl",
        "loop_decision.jsonl",
    ]
    for name in native_files:
        for line in (BATCH2 / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            assert rec.get("provider_capability") != "context"


def test_development_pack_can_format_context_fixtures() -> None:
    records = generate_context_records()
    examples = format_fp2_pack(records, pack="development")
    assert len(examples) == len(records)
    assert all(ex.dataset_type.value == "context" for ex in examples)


def test_forbidden_how_still_blocked_on_context_path() -> None:
    with pytest.raises(Exception):
        assert_no_forbidden_how(capability_id="local_workspace")
