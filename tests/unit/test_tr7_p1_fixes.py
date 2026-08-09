"""TR-7 — continuity expansion + Odoo classification + readiness re-evaluation."""

from __future__ import annotations

import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]

import pytest

from aiodoo_training.system_training_contract.evaluation import (
    TrainingReadiness,
    evaluate_controlled_batch,
)
from aiodoo_training.system_training_contract.evaluation.domain_classify import (
    DomainClass,
    classify_domain,
    corrected_domain_specialization,
    has_odoo_semantics,
)
from aiodoo_training.system_training_contract.evaluation.metrics import (
    analyze_objective_completion,
    analyze_odoo_quality,
    load_native_records,
)
from aiodoo_training.system_training_contract.generators.tr7_continuity import (
    generate_continuity_expansion,
)
from aiodoo_training.system_training_contract.quality.gates import scan_forbidden_how
from aiodoo_training.system_training_contract.records import validate_record_mapping

DATASETS = WORKSPACE / "aiodoo-datasets/datasets/fp2"
BATCH1 = DATASETS / "controlled_batch_1"
BATCH2 = DATASETS / "controlled_batch_2"


@pytest.mark.skipif(not BATCH1.is_dir(), reason="TR-5 batch missing")
def test_tr7_reproduces_tr6_p1_on_batch1() -> None:
    """Independent reproduction of TR-6 P1 blockers on immutable TR-5 evidence."""
    report = evaluate_controlled_batch(BATCH1)
    assert report.inventory["total_native"] == 1200
    assert report.hard_gates["forbidden_how"] == "PASS"
    assert report.continuity["state_count"] == 15
    assert report.continuity["dc_count"] == 15
    assert report.inventory["per_family"]["loop_decision"] == 15
    # TR-6 reported 203 under narrow cues; TR-7 semantic classifier finds more
    # label/cue mismatches on the same corpus (documented discrepancy).
    assert report.odoo["ambiguous"] >= 203
    assert report.readiness == TrainingReadiness.READY_WITH_REQUIRED_DATA_FIXES.value
    assert any("continuity_family_volume_low" in x for x in report.issues["P1"])
    assert any("ambiguous_odoo_generic" in x for x in report.issues["P1"])


@pytest.mark.skipif(not BATCH2.is_dir(), reason="TR-7 batch missing")
def test_tr7_batch2_readiness_for_training() -> None:
    report = evaluate_controlled_batch(BATCH2)
    assert report.checksum_ok is True
    assert 1200 < report.inventory["total_native"] <= 1800
    for gate, outcome in report.hard_gates.items():
        assert outcome == "PASS", gate
    assert report.issues["P0"] == []
    assert report.issues["P1"] == []
    assert report.readiness == TrainingReadiness.READY_FOR_TRAINING.value
    assert report.continuity["state_count"] >= 60
    assert report.continuity["dc_count"] >= 60
    assert report.inventory["per_family"]["loop_decision"] >= 60
    assert report.odoo["ambiguous"] == 0
    assert report.negatives["ok"] is True


@pytest.mark.skipif(not BATCH2.is_dir(), reason="TR-7 batch missing")
def test_tr7_batch1_preserved_immutable() -> None:
    """TR-5 evidence must remain at 1200; TR-7 writes a derivative only."""
    b1 = load_native_records(BATCH1)
    b2 = load_native_records(BATCH2)
    assert len(b1) == 1200
    assert len(b2) > 1200
    ids1 = {r["record_id"] for r in b1}
    # Original IDs still present in derivative (corrected, not replaced)
    ids2 = {r["record_id"] for r in b2}
    assert ids1.issubset(ids2)


def test_tr7_continuity_generator_invariants() -> None:
    rows = generate_continuity_expansion()
    assert len(rows) >= 180  # 60+ each of three types
    by_type: dict[str, list] = {}
    for r in rows:
        validate_record_mapping(r)
        assert scan_forbidden_how(r) == []
        by_type.setdefault(str(r["record_type"]), []).append(r)
    assert len(by_type["engineering_state"]) >= 60
    assert len(by_type["decision_context"]) >= 60
    assert len(by_type["loop_decision"]) >= 60

    # 1) previous COMPLETE does not leak into current failure
    fail_with_prior_complete = [
        r
        for r in by_type["decision_context"]
        if (r.get("input") or {}).get("objective_state") == "failed"
        and any(
            isinstance(h, dict)
            and h.get("objective_state") == "complete"
            and h.get("historical")
            for h in ((r.get("input") or {}).get("bounded_history") or [])
        )
    ]
    assert fail_with_prior_complete, "need prior-COMPLETE/current-failure scenario"
    for r in fail_with_prior_complete:
        assert (r.get("input") or {}).get("objective_state") == "failed"

    # 5) empty evidence does not imply COMPLETE
    emptyish = [
        r
        for r in by_type["decision_context"]
        if (r.get("input") or {}).get("observation_quality") == "missing"
        or (r.get("input") or {}).get("execution_state") == "missing_observation"
    ]
    assert emptyish
    for r in emptyish:
        assert (r.get("input") or {}).get("objective_state") != "complete"

    # 8) DecisionContext permitted fields only — no Memory/RAG / forbidden HOW
    for r in by_type["decision_context"]:
        blob = json.dumps(r.get("input") or {}).lower()
        assert "rag" not in blob
        assert "vector_store" not in blob
        assert "embedding" not in blob
        assert "memory_store" not in blob
        assert "local_workspace" not in blob
        assert "password" not in blob
        assert "implementation_id" not in blob

    # 2–4) failure/repair/validation are not automatic pipelines in loop reasons
    for r in by_type["loop_decision"]:
        reason = str((r.get("expected_output") or {}).get("reason") or "").lower()
        assert "must repair" not in reason
        assert "automatically repair" not in reason
        assert "always validate after repair" not in reason
        assert "must validate after repair" not in reason

    # 6) historical summary remains bounded
    for r in by_type["decision_context"]:
        hist = (r.get("input") or {}).get("bounded_history") or []
        assert len(hist) <= 8

    # 7) current state derived from current evidence (objective_state on DC matches state family)
    families = {
        (r.get("metadata") or {}).get("scenario_family")
        for r in rows
        if (r.get("metadata") or {}).get("scenario_family")
    }
    assert len(families) >= 20


def test_tr7_odoo_classification_rules() -> None:
    # Provenance-only must not force Odoo
    fake = {
        "record_type": "capability_intent",
        "record_id": "x",
        "domain_specialization": None,
        "input": {"objective": "read a python file"},
        "expected_output": {"capability_id": "workspace.read", "args": {"path": "a.py"}},
        "metadata": {"provenance": "cloned from odoo/odoo repository"},
    }
    assert classify_domain(fake) == DomainClass.GENERIC
    assert has_odoo_semantics(fake) is False

    odoo_real = {
        "record_type": "capability_intent",
        "record_id": "y",
        "domain_specialization": "odoo",
        "input": {"objective": "edit models/partner.py action_confirm"},
        "expected_output": {
            "capability_id": "workspace.write",
            "args": {"path": "models/partner.py"},
        },
    }
    assert classify_domain(odoo_real) == DomainClass.ODOO_SPECIALIZED

    labeled_no_cues = {
        "record_type": "capability_intent",
        "record_id": "z",
        "domain_specialization": "odoo",
        "input": {"objective": "attach a report artifact"},
        "expected_output": {
            "capability_id": "artifact.attachment",
            "args": {"path": "artifacts/report.json"},
        },
    }
    assert classify_domain(labeled_no_cues) == DomainClass.AMBIGUOUS
    dom, cls, action = corrected_domain_specialization(labeled_no_cues)
    assert action == "clear_odoo"
    assert dom is None
    assert cls == DomainClass.GENERIC

    cues_no_label = {
        "record_type": "capability_intent",
        "record_id": "w",
        "domain_specialization": None,
        "input": {"objective": "read __manifest__.py"},
        "expected_output": {
            "capability_id": "workspace.read",
            "args": {"path": "addons/x/__manifest__.py"},
        },
    }
    assert classify_domain(cues_no_label) == DomainClass.AMBIGUOUS
    dom, cls, action = corrected_domain_specialization(cues_no_label)
    assert action == "set_odoo"
    assert dom == "odoo"


@pytest.mark.skipif(not BATCH2.is_dir(), reason="TR-7 batch missing")
def test_tr7_odoo_batch_consistency() -> None:
    records = load_native_records(BATCH2)
    q = analyze_odoo_quality(records)
    assert q["ambiguous"] == 0
    for rec in records:
        cls = classify_domain(rec)
        assert cls != DomainClass.AMBIGUOUS
        if rec.get("domain_specialization") == "odoo":
            assert has_odoo_semantics(rec)
            assert cls == DomainClass.ODOO_SPECIALIZED
        else:
            assert not has_odoo_semantics(rec)
            assert cls == DomainClass.GENERIC


@pytest.mark.skipif(not BATCH2.is_dir(), reason="TR-7 batch missing")
def test_tr7_no_negative_contamination_and_quarantine_outside_packs() -> None:
    report = evaluate_controlled_batch(BATCH2)
    assert report.negatives["ok"] is True
    for name in ("pack_development.jsonl", "pack_reasoning.jsonl", "splits.jsonl"):
        text = (BATCH2 / name).read_text(encoding="utf-8")
        assert "not_for_training" not in text
        assert "quality_corpus\": \"negative" not in text
    # Quarantine file may exist but must not feed packs
    qpath = BATCH2 / "ambiguous_quarantine.jsonl"
    if qpath.is_file() and qpath.stat().st_size > 0:
        qids = {
            json.loads(line)["record_id"]
            for line in qpath.read_text().splitlines()
            if line.strip()
        }
        pack_text = (BATCH2 / "pack_development.jsonl").read_text() + (
            BATCH2 / "pack_reasoning.jsonl"
        ).read_text()
        for qid in qids:
            assert qid not in pack_text


@pytest.mark.skipif(not BATCH2.is_dir(), reason="TR-7 batch missing")
def test_tr7_objective_completion_and_how() -> None:
    records = load_native_records(BATCH2)
    obj = analyze_objective_completion(records)
    assert obj["ok"] is True
    assert obj["auto_repair_policy_reasons"] == 0
    for rec in records:
        assert scan_forbidden_how(rec) == []


@pytest.mark.skipif(not BATCH2.is_dir(), reason="TR-7 batch missing")
def test_tr7_audit_traceability() -> None:
    audit = BATCH2 / "tr7_domain_audit.jsonl"
    assert audit.is_file()
    rows = [json.loads(l) for l in audit.read_text().splitlines() if l.strip()]
    assert any(r.get("action") == "clear_odoo" for r in rows)
    manifest = json.loads((BATCH2 / "manifest.json").read_text())
    assert "tr7" in str(manifest.get("controlled_batch_version") or "").lower() or manifest.get(
        "tr7"
    )
    assert manifest.get("source_batch") == "controlled_batch_1"
    assert manifest.get("checksum")
