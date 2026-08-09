"""AT-6.2.1 — Controlled FP2 Context corpus expansion & readiness."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from aiodoo_training.system_training_contract.generators.context import (
    CONTEXT_CORPUS_VERSION,
    generate_context_records,
)
from aiodoo_training.system_training_contract.generators.context_controlled import (
    BATCH2_CHECKSUM,
    CONTEXT_CONTROLLED_VERSION,
    CONTEXT_LOCATE_CAPS,
    TARGET_MAX,
    TARGET_MIN,
    analyze_context_controlled,
    emit_context_controlled_corpus,
    find_normalized_duplicates,
    generate_context_controlled_records,
)
from aiodoo_training.system_training_contract.generators.mapping import (
    CONTEXT_ALLOWED_RECORD_TYPES,
    record_provider_capabilities,
)
from aiodoo_training.system_training_contract.quality.analysis import find_duplicates
from aiodoo_training.system_training_contract.quality.gates import scan_forbidden_how
from aiodoo_training.system_training_contract.quality.splits import assign_split
from aiodoo_training.system_training_contract.records import validate_record_mapping

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]
BATCH2 = WORKSPACE / "aiodoo-datasets" / "datasets" / "fp2" / "controlled_batch_2"
AT62 = WORKSPACE / "aiodoo-training" / "fixtures" / "fp2" / "context"
CONTROLLED_TRAIN = WORKSPACE / "aiodoo-training" / "fixtures" / "fp2" / "context_controlled_1"


def test_at62_fixtures_untouched() -> None:
    assert AT62.is_dir()
    man = json.loads((AT62 / "manifest.json").read_text(encoding="utf-8"))
    assert man["version"] == CONTEXT_CORPUS_VERSION == "fp2-context-1.0.0"
    assert man["total_records"] == 26
    assert len(generate_context_records()) == 26


def test_controlled_count_and_determinism() -> None:
    first = generate_context_controlled_records()
    second = generate_context_controlled_records()
    assert TARGET_MIN <= len(first) <= TARGET_MAX
    assert [json.dumps(r, sort_keys=True) for r in first] == [
        json.dumps(r, sort_keys=True) for r in second
    ]
    assert len({r["record_id"] for r in first}) == len(first)


def test_record_type_and_mapping() -> None:
    records = generate_context_controlled_records()
    by_type = Counter(r["record_type"] for r in records)
    assert set(by_type) <= set(CONTEXT_ALLOWED_RECORD_TYPES)
    assert by_type["capability_intent"] >= 60
    assert by_type["observation"] >= 60
    for r in records:
        assert r["provider_capability"] == "context"
        assert "context" in record_provider_capabilities(r["record_type"])
        validate_record_mapping(r)


def test_capability_coverage() -> None:
    records = generate_context_controlled_records()
    caps: Counter[str] = Counter()
    for r in records:
        if r["record_type"] == "capability_intent":
            caps[r["expected_output"]["capability_id"]] += 1
        else:
            caps[r["evidence"]["capability_id"]] += 1
    for cap in CONTEXT_LOCATE_CAPS:
        assert caps[cap] >= 10, cap
    assert set(caps) <= set(CONTEXT_LOCATE_CAPS)


def test_odoo_generic_and_families() -> None:
    records = generate_context_controlled_records()
    odoo = sum(1 for r in records if r.get("domain_specialization") == "odoo")
    generic = len(records) - odoo
    assert odoo >= 40 and generic >= 40
    families = {r["metadata"]["scenario_family"] for r in records}
    assert len(families) >= 40
    for r in records:
        assert r["metadata"]["scenario_family"]
        assert r["metadata"].get("legacy") is False


def test_duplicates_and_forbidden_how() -> None:
    records = generate_context_controlled_records()
    exact = find_duplicates(records)
    assert exact["duplicate_groups"] == 0
    norm = find_normalized_duplicates(records)
    assert norm["normalized_duplicate_groups"] == 0
    assert all(not scan_forbidden_how(r) for r in records)


def test_scenario_family_isolation() -> None:
    records = generate_context_controlled_records()
    family_splits: dict[str, set[str]] = {}
    for r in records:
        fam = r["metadata"]["scenario_family"]
        family_splits.setdefault(fam, set()).add(assign_split(r).value)
    assert all(len(v) == 1 for v in family_splits.values())


def test_negatives_excluded_from_corpus() -> None:
    records = generate_context_controlled_records()
    assert all(
        not str(r["metadata"].get("quality_corpus") or "").startswith("negative")
        for r in records
    )


def test_analyze_ready_and_pack() -> None:
    records = generate_context_controlled_records()
    report = analyze_context_controlled(records)
    assert report["verdict"] == "CONTEXT_CORPUS_READY"
    assert report["scorecard"]["pack_validity"] is True
    assert report["scorecard"]["family_leakage"] == 0
    assert report["pack_count"] == len(records)


def test_emit_and_batch2_immutable(tmp_path: Path) -> None:
    before = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))["checksum"]
    assert before == BATCH2_CHECKSUM
    result = emit_context_controlled_corpus(
        training_root=tmp_path / "train",
        datasets_root=tmp_path / "data",
    )
    assert result["version"] == CONTEXT_CONTROLLED_VERSION
    assert result["verdict"] == "CONTEXT_CORPUS_READY"
    assert (tmp_path / "data" / "pack_context.jsonl").is_file()
    assert (tmp_path / "data" / "splits.jsonl").is_file()
    assert (tmp_path / "data" / "quality_report.json").is_file()
    after = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))["checksum"]
    assert after == before == BATCH2_CHECKSUM
    assert not any("controlled_batch_2" in p for p in result["written"])
    # AT-6.2 fixtures still 26
    assert json.loads((AT62 / "manifest.json").read_text(encoding="utf-8"))["total_records"] == 26


def test_on_disk_controlled_corpus_present() -> None:
    assert CONTROLLED_TRAIN.is_dir()
    man = json.loads((CONTROLLED_TRAIN / "manifest.json").read_text(encoding="utf-8"))
    assert man["version"] == CONTEXT_CONTROLLED_VERSION
    assert TARGET_MIN <= man["total_records"] <= TARGET_MAX
    assert man["verdict"] == "CONTEXT_CORPUS_READY"
    assert man["controlled_batch_2_modified"] is False
    assert man["at62_fixtures_modified"] is False
