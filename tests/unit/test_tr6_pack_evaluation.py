"""TR-6 — FP2 corpus evaluation / training-pack readiness tests."""

from __future__ import annotations

import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]

import pytest

from aiodoo_training.system_training_contract.evaluation import (
    TrainingReadiness,
    evaluate_controlled_batch,
    render_tr6_scorecard,
)
from aiodoo_training.system_training_contract.evaluation.metrics import (
    analyze_objective_completion,
    analyze_scenario_diversity,
    load_native_records,
    normalize_scenario_family,
)

BATCH = WORKSPACE / "aiodoo-datasets/datasets/fp2/controlled_batch_1"


@pytest.mark.skipif(not BATCH.is_dir(), reason="controlled batch missing")
def test_tr6_evaluation_readiness_and_hard_gates() -> None:
    report = evaluate_controlled_batch(BATCH)
    assert report.inventory["total_native"] == 1200
    assert report.checksum_ok is True
    assert report.inventory["development_pack"] == 942
    assert report.inventory["reasoning_pack"] == 954
    for gate, outcome in report.hard_gates.items():
        assert outcome == "PASS", gate
    assert report.readiness == TrainingReadiness.READY_WITH_REQUIRED_DATA_FIXES.value
    assert report.issues["P0"] == []
    assert any("continuity_family_volume_low" in x for x in report.issues["P1"])
    text = render_tr6_scorecard(report)
    assert "Readiness: READY_WITH_REQUIRED_DATA_FIXES" in text


@pytest.mark.skipif(not BATCH.is_dir(), reason="controlled batch missing")
def test_tr6_diversity_and_objective_semantics() -> None:
    records = load_native_records(BATCH)
    div = analyze_scenario_diversity(records)
    assert div["unique_families"] >= 40
    assert div["concentration_pct"] < 15
    assert normalize_scenario_family("nav_tests_r2") == "nav_tests"
    obj = analyze_objective_completion(records)
    assert obj["ok"] is True
    assert obj["operation_success_objective_incomplete"] >= 1
    assert obj["auto_repair_policy_reasons"] == 0


@pytest.mark.skipif(not BATCH.is_dir(), reason="controlled batch missing")
def test_tr6_report_artifact_exists_after_eval() -> None:
    report = evaluate_controlled_batch(BATCH)
    path = BATCH / "quality_report_tr6.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    data = json.loads(path.read_text())
    assert data["readiness"] == "READY_WITH_REQUIRED_DATA_FIXES"
    assert data["hard_gates"]["forbidden_how"] == "PASS"
    assert data["splits"]["leakage_count"] == 0
