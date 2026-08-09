"""AT-7.5 — Controlled Evaluation FP2 corpus generation & readiness."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from aiodoo_training.bootstrap import bootstrap_phase7
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.refs import RECORD_FORMAT_FP2_TRAINING_EXAMPLE, DatasetRef
from aiodoo_training.system_training_contract.generators.evaluation_controlled import (
    BATCH2_CHECKSUM,
    EVALUATION_CONTROLLED_VERSION,
    TARGET_MAX,
    TARGET_MIN,
    analyze_evaluation_controlled,
    emit_evaluation_controlled_corpus,
    find_normalized_duplicates,
    generate_evaluation_controlled_records,
)
from aiodoo_training.system_training_contract.generators.mapping import (
    EVALUATION_ALLOWED_RECORD_TYPES,
    record_provider_capabilities,
)
from aiodoo_training.system_training_contract.quality.analysis import find_duplicates
from aiodoo_training.system_training_contract.quality.gates import scan_forbidden_how
from aiodoo_training.system_training_contract.quality.negatives import (
    REASONING_SPARSE_NEGATIVE_CASES,
    evaluate_negative_case,
)
from aiodoo_training.system_training_contract.quality.splits import assign_split
from aiodoo_training.system_training_contract.records import validate_record_mapping

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]
BATCH2 = WORKSPACE / "aiodoo-datasets" / "datasets" / "fp2" / "controlled_batch_2"
EVAL_TRAIN = (
    WORKSPACE / "aiodoo-training" / "fixtures" / "fp2" / "reasoning_controlled_1" / "evaluation"
)
EVAL_DATA = (
    WORKSPACE / "aiodoo-datasets" / "datasets" / "fp2" / "reasoning_controlled_1" / "evaluation"
)
CONV_MAN = (
    WORKSPACE
    / "aiodoo-datasets"
    / "datasets"
    / "fp2"
    / "reasoning_controlled_1"
    / "conversation"
    / "manifest.json"
)
APPR_MAN = (
    WORKSPACE
    / "aiodoo-datasets"
    / "datasets"
    / "fp2"
    / "reasoning_controlled_1"
    / "approval"
    / "manifest.json"
)


def test_count_determinism_and_ids() -> None:
    first = generate_evaluation_controlled_records()
    second = generate_evaluation_controlled_records()
    assert TARGET_MIN <= len(first) <= TARGET_MAX
    assert [json.dumps(r, sort_keys=True) for r in first] == [
        json.dumps(r, sort_keys=True) for r in second
    ]
    assert len({r["record_id"] for r in first}) == len(first)


def test_schema_provider_and_family() -> None:
    records = generate_evaluation_controlled_records()
    assert {r["record_type"] for r in records} == {"evaluation_judgment"}
    assert EVALUATION_ALLOWED_RECORD_TYPES == frozenset({"evaluation_judgment"})
    for r in records:
        assert r["provider_capability"] == "evaluation"
        assert "evaluation" in record_provider_capabilities(r["record_type"])
        validate_record_mapping(r)
        assert r["training_contract_version"] == "1.0.0"


def test_verdict_score_and_optional_fields() -> None:
    records = generate_evaluation_controlled_records()
    verdicts = Counter(r["expected_output"]["verdict"] for r in records)
    assert verdicts["pass"] and verdicts["fail"] and verdicts["inconclusive"]
    with_score = sum(1 for r in records if "score" in r["expected_output"])
    without_score = len(records) - with_score
    assert with_score and without_score
    for r in records:
        if "score" in r["expected_output"]:
            s = float(r["expected_output"]["score"])
            assert 0.0 <= s <= 1.0
    patterns = Counter()
    for r in records:
        inp = r["input"]
        has_e = "expectation" in inp
        has_r = "rubric" in inp
        if has_e and has_r:
            patterns["all"] += 1
        elif has_e:
            patterns["exp"] += 1
        elif has_r:
            patterns["rub"] += 1
        else:
            patterns["cand"] += 1
    assert patterns["all"] and patterns["exp"] and patterns["rub"] and patterns["cand"]


def test_families_splits_and_isolation() -> None:
    records = generate_evaluation_controlled_records()
    families = {r["metadata"]["scenario_family"] for r in records}
    assert len(families) >= 50
    family_splits: dict[str, set[str]] = {}
    splits = Counter()
    for r in records:
        fam = r["metadata"]["scenario_family"]
        split = assign_split(r).value
        family_splits.setdefault(fam, set()).add(split)
        splits[split] += 1
    assert all(len(v) == 1 for v in family_splits.values())
    assert splits["validation"] > 0 and splits["test"] > 0


def test_duplicates_how_legacy_negatives() -> None:
    records = generate_evaluation_controlled_records()
    assert find_duplicates(records)["duplicate_groups"] == 0
    assert find_normalized_duplicates(records)["normalized_duplicate_groups"] == 0
    assert all(not scan_forbidden_how(r) for r in records)
    assert all(r["metadata"].get("legacy") is False for r in records)
    assert all(r["metadata"].get("legacy_projection") is False for r in records)
    blob = json.dumps(records)
    assert "evaluation_dataset.jsonl" not in blob
    assert all(evaluate_negative_case(c)["matched"] for c in REASONING_SPARSE_NEGATIVE_CASES)
    assert all(
        not str(r["metadata"].get("quality_corpus") or "").startswith("negative")
        for r in records
    )


def test_analyze_ready_and_pack() -> None:
    report = analyze_evaluation_controlled(generate_evaluation_controlled_records())
    assert report["verdict"] == "EVALUATION_CORPUS_READY"
    assert report["scorecard"]["pack_validity"] is True
    assert report["scorecard"]["provider_dataset_equivalence"] is True
    assert report["scorecard"]["family_leakage"] == 0
    assert report["legacy_projection"] == "NOT PERFORMED"


def test_emit_loader_and_immutability(tmp_path: Path) -> None:
    before = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))["checksum"]
    assert before == BATCH2_CHECKSUM
    conv_before = json.loads(CONV_MAN.read_text(encoding="utf-8"))
    appr_before = json.loads(APPR_MAN.read_text(encoding="utf-8"))

    result = emit_evaluation_controlled_corpus(
        training_root=tmp_path / "train",
        datasets_root=tmp_path / "data",
    )
    assert result["verdict"] == "EVALUATION_CORPUS_READY"
    assert result["version"] == EVALUATION_CONTROLLED_VERSION
    assert result["legacy_projection"] == "NOT PERFORMED"
    pack = tmp_path / "data" / "pack_evaluation.jsonl"
    assert pack.is_file()
    assert (tmp_path / "data" / "evaluation_judgment.jsonl").is_file()
    assert (tmp_path / "data" / "splits.jsonl").is_file()

    bootstrap_phase7(overwrite=True)
    ref = DatasetRef(
        path=pack,
        dataset_type=DatasetType.EVALUATION,
        protocol_version="1.0.0",
        record_format=RECORD_FORMAT_FP2_TRAINING_EXAMPLE,
        name="eval_pack",
    )
    examples = list(JsonlDatasetSource(validate=True).load([ref]))
    assert len(examples) == result["count"]
    assert all(ex.dataset_type == DatasetType.EVALUATION for ex in examples)
    assert all(ex.metadata.get("provider_capability") == "evaluation" for ex in examples)
    assert all(ex.metadata.get("record_type") == "evaluation_judgment" for ex in examples)

    after = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))["checksum"]
    assert after == before == BATCH2_CHECKSUM
    assert json.loads(CONV_MAN.read_text(encoding="utf-8")) == conv_before
    assert json.loads(APPR_MAN.read_text(encoding="utf-8")) == appr_before
    assert not any("controlled_batch_2" in p for p in result["written"])


def test_on_disk_corpus_and_semantics_preserved() -> None:
    assert EVAL_TRAIN.is_dir() and EVAL_DATA.is_dir()
    for root in (EVAL_TRAIN, EVAL_DATA):
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert man["version"] == EVALUATION_CONTROLLED_VERSION
        assert man["verdict"] == "EVALUATION_CORPUS_READY"
        assert TARGET_MIN <= man["total_records"] <= TARGET_MAX
        assert man["legacy_projection_status"] == "NOT PERFORMED"
        assert (root / "semantics_report.json").is_file()
        assert (root / "pack_evaluation.jsonl").is_file()
        assert (root / "evaluation_native.jsonl").is_file()
    assert json.loads(CONV_MAN.read_text(encoding="utf-8"))["total_records"] == 232
    assert json.loads(APPR_MAN.read_text(encoding="utf-8"))["total_records"] == 162
