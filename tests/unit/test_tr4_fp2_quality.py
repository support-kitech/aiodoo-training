"""TR-4 — FP2 corpus quality harness tests."""

from __future__ import annotations

import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]

import pytest

from aiodoo_training.system_training_contract.quality import (
    SplitAssignment,
    assign_split,
    document_split_strategy,
    evaluate_fp2_corpus,
    render_scorecard,
)
from aiodoo_training.system_training_contract.quality.formatters import (
    format_fp2_pack,
    format_fp2_record,
)
from aiodoo_training.system_training_contract.quality.gates import (
    scan_forbidden_how,
    scan_taxonomy,
)
from aiodoo_training.system_training_contract.quality.negatives import (
    NEGATIVE_CASES,
    evaluate_negative_case,
)
from aiodoo_training.system_training_contract.records import validate_record_mapping
from aiodoo_training.system_training_contract.taxonomy import (
    PREFERRED_ENGINEERING_CAPABILITY_IDS,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "fp2"


def test_fixture_inventory_111() -> None:
    from aiodoo_training.system_training_contract.quality.common import NATIVE_FAMILIES

    native = 0
    for fam in NATIVE_FAMILIES:
        native += sum(1 for _ in (FIXTURES / f"{fam}.jsonl").open())
    proj = sum(1 for _ in (FIXTURES / "projection_fixtures.jsonl").open())
    # TR-5.2 expanded fixtures for coverage gaps (was 104 native / 111 total).
    assert native >= 104
    assert proj == 7
    assert native + proj >= 111
    # Negatives exist but are quality-only (not part of training fixtures)
    assert (FIXTURES / "quality_negatives.jsonl").is_file()


def test_quality_harness_ready() -> None:
    report = evaluate_fp2_corpus(FIXTURES)
    assert report.total_native_records >= 104
    assert report.total_projection_records == 7
    assert report.gates["schema"] == "PASS"
    assert report.gates["forbidden_how"] == "PASS"
    assert report.gates["taxonomy"] == "PASS"
    assert report.gates["continuity"] == "PASS"
    assert report.gates["negative_corpus"] == "PASS"
    # Inventory gate still expects classic 104; after TR-5.2 expansion it may WARN.
    assert report.gates["schema"] == "PASS"
    assert report.duplicates.get("duplicate_groups") == 0
    assert report.coverage.get("coverage_pct", 0) >= 95
    text = render_scorecard(report)
    assert "Gates:" in text


def test_forbidden_how_and_taxonomy_negative_positive() -> None:
    bad = {
        "record_type": "capability_intent",
        "provider_capability": "planner",
        "expected_output": {"capability_id": "local_git", "args": {"command": "ls"}},
        "input": {"objective": "x"},
    }
    assert scan_forbidden_how(bad)
    confused = {
        "record_type": "capability_intent",
        "provider_capability": "workspace.write",
        "expected_output": {"capability_id": "coding", "args": {}},
        "input": {"objective": "x"},
    }
    assert scan_taxonomy(confused)


def test_historical_meta_how_not_flagged_in_model_facing() -> None:
    ok = {
        "record_type": "capability_intent",
        "provider_capability": "planner",
        "input": {"objective": "Write file", "reason": "ok"},
        "expected_output": {"capability_id": "workspace.write", "args": {"path": "a.py"}},
        "provenance": {"notes": "legacy local_workspace create_file"},
        "metadata": {"historical_note": "local_git"},
    }
    # Model-facing scan should not read provenance/metadata
    assert not scan_forbidden_how(ok)


def test_coverage_matrix_uses_preferred_taxonomy() -> None:
    report = evaluate_fp2_corpus(FIXTURES)
    assert report.coverage["preferred_total"] == len(PREFERRED_ENGINEERING_CAPABILITY_IDS)
    assert report.coverage["coverage_pct"] >= 95
    assert report.coverage["uncovered"] == []


def test_work_planning_feedback_loop_gates() -> None:
    report = evaluate_fp2_corpus(FIXTURES)
    assert report.work_units["ok"]
    assert report.planning["ok"]
    assert report.feedback["ok"]
    assert report.loops["ok"]
    assert report.continuity["ok"]


def test_odoo_generic_and_splits() -> None:
    report = evaluate_fp2_corpus(FIXTURES)
    assert report.domain["odoo"] > 0
    assert report.domain["generic"] > 0
    strategy = document_split_strategy()
    assert strategy["ratios"]["train"] == 0.80
    # Multi-cycle scenarios share family keys
    c1 = json.loads((FIXTURES / "engineering_state.jsonl").read_text().splitlines()[0])
    # find cycle scenarios
    states = [
        json.loads(line) for line in (FIXTURES / "engineering_state.jsonl").read_text().splitlines()
    ]
    by_scene = {r["metadata"]["scenario"]: r for r in states}
    s1 = assign_split(by_scene["cycle1_validation_failed"])
    s2 = assign_split(by_scene["cycle2_repair_applied"])
    s3 = assign_split(by_scene["cycle3_validation_passed"])
    assert s1 == s2 == s3
    assert isinstance(s1, SplitAssignment)


def test_negative_corpus_all_match() -> None:
    results = [evaluate_negative_case(c) for c in NEGATIVE_CASES]
    assert all(r["matched"] for r in results)
    assert any(r["case_id"].startswith("pos_") and r["accepted"] for r in results)
    assert any(r["case_id"].startswith("neg_") and not r["accepted"] for r in results)


def test_fp2_formatters_preserve_semantics() -> None:
    line = (FIXTURES / "capability_intent.jsonl").read_text().splitlines()[0]
    rec = json.loads(line)
    validate_record_mapping(rec)
    ex = format_fp2_record(rec)
    assert ex.messages[0]["role"] == "user"
    assert ex.messages[1]["role"] == "assistant"
    assert "local_" not in ex.messages[1]["content"] or "local_" not in json.loads(
        ex.messages[1]["content"]
    ).get("capability_id", "")
    assert ex.metadata["fp2_native"] is True
    records = [json.loads(l) for l in (FIXTURES / "planning_decision.jsonl").read_text().splitlines()]
    reasoning = format_fp2_pack(records, pack="reasoning")
    assert reasoning
    with pytest.raises(ValueError):
        format_fp2_pack(records, pack="both")


def test_legacy_production_untouched() -> None:
    import hashlib

    root = Path(__file__).resolve().parents[3].parent / "aiodoo-datasets" / "datasets"
    # workspace is aidevelopment; training repo parent is aidevelopment
    datasets = WORKSPACE / "aiodoo-datasets/datasets"
    expected = {
        "planner_v1_0.jsonl": "5a0165685d5360acc4648db343b82731",
        "coding_v1_0.jsonl": "e85fee07943b71271f4408e709a964ee",
        "execution_dataset.jsonl": "092102a5c048ffa0e13f994541bc77f6",
    }
    for name, digest in expected.items():
        data = (datasets / name).read_bytes()
        assert hashlib.md5(data).hexdigest() == digest
