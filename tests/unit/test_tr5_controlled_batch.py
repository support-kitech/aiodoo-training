"""TR-5 — Controlled FP2 batch generation tests."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]

import pytest

from aiodoo_training.system_training_contract.generators.controlled_batch import (
    CONTROLLED_BATCH_MAX,
    CONTROLLED_BATCH_MIN,
    CONTROLLED_BATCH_VERSION,
    emit_controlled_batch,
    generate_controlled_batch_records,
)
from aiodoo_training.system_training_contract.quality.analysis import capability_coverage
from aiodoo_training.system_training_contract.quality.gates import (
    scan_forbidden_how,
    scan_taxonomy,
)
from aiodoo_training.system_training_contract.quality.splits import assign_split
from aiodoo_training.system_training_contract.taxonomy import PREFERRED_ENGINEERING_CAPABILITY_IDS

BATCH = WORKSPACE / "aiodoo-datasets/datasets/fp2/controlled_batch_1"
DATASETS = WORKSPACE / "aiodoo-datasets/datasets"


@pytest.mark.skipif(not BATCH.is_dir(), reason="controlled batch not generated")
def test_controlled_batch_inventory_and_coverage() -> None:
    report = json.loads((BATCH / "quality_report_tr5.json").read_text(encoding="utf-8"))
    assert report["decision"] == "PASS"
    assert CONTROLLED_BATCH_MIN <= report["total_native"] <= CONTROLLED_BATCH_MAX
    assert report["coverage"]["coverage_pct"] == 100.0
    assert report["coverage"]["uncovered"] == []
    assert set(report["per_capability"]) == set(PREFERRED_ENGINEERING_CAPABILITY_IDS)
    assert report["duplicates"]["duplicate_groups"] == 0
    assert report["quality"]["how_violations"] == []
    assert report["quality"]["taxonomy_violations"] == []
    assert report["controlled_batch_version"] == CONTROLLED_BATCH_VERSION


@pytest.mark.skipif(not BATCH.is_dir(), reason="controlled batch not generated")
def test_controlled_batch_splits_and_no_negative_contamination() -> None:
    report = json.loads((BATCH / "quality_report_tr5.json").read_text(encoding="utf-8"))
    splits = report["split_counts"]
    total = sum(splits.values())
    assert total == report["total_native"]
    assert splits["train"] / total >= 0.70
    # packs
    for name in ("pack_development.jsonl", "pack_reasoning.jsonl", "splits.jsonl"):
        text = (BATCH / name).read_text(encoding="utf-8")
        assert "not_for_training" not in text
        assert 'quality_corpus": "negative' not in text
    # narrative family leakage
    by_fam: dict[str, set[str]] = defaultdict(set)
    for line in (BATCH / "engineering_state.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        fam = (rec.get("metadata") or {}).get("scenario_family")
        if fam and str(fam).startswith("narrative_"):
            by_fam[str(fam)].add(assign_split(rec).value)
    for fam, sps in by_fam.items():
        assert len(sps) == 1, fam


def test_generate_controlled_batch_records_in_memory() -> None:
    records = generate_controlled_batch_records(target=200)
    assert 150 <= len(records) <= 200
    cov = capability_coverage(records)
    assert cov["coverage_pct"] == 100.0
    for rec in records[:20]:
        assert not scan_forbidden_how(rec)
        assert not scan_taxonomy(rec)
        meta = rec.get("metadata") or {}
        assert not str(meta.get("quality_corpus") or "").startswith("negative")


def test_emit_controlled_batch_tmp(tmp_path: Path) -> None:
    result = emit_controlled_batch(tmp_path / "cb", target=200)
    assert result.decision in {"PASS", "PASS_WITH_WARNINGS"}
    assert result.total_native >= 150
    assert (tmp_path / "cb" / "manifest.json").is_file()
    assert (tmp_path / "cb" / "pack_development.jsonl").is_file()
    assert (tmp_path / "cb" / "pack_reasoning.jsonl").is_file()
    assert result.coverage.get("uncovered") == []


def test_legacy_production_still_untouched() -> None:
    expected = {
        "planner_v1_0.jsonl": "5a0165685d5360acc4648db343b82731",
        "coding_v1_0.jsonl": "e85fee07943b71271f4408e709a964ee",
        "execution_dataset.jsonl": "092102a5c048ffa0e13f994541bc77f6",
    }
    for name, digest in expected.items():
        assert hashlib.md5((DATASETS / name).read_bytes()).hexdigest() == digest
