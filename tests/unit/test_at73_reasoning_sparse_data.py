"""AT-7.3 — Reasoning sparse-skill Conversation / Approval data readiness."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from aiodoo_training.system_training_contract.generators.approval_controlled import (
    APPROVAL_CONTROLLED_GENERATOR,
    MIN_FAMILIES as APPROVAL_MIN_FAMILIES,
    REASONING_SPARSE_VERSION,
    TARGET_MAX as APPROVAL_TARGET_MAX,
    TARGET_MIN as APPROVAL_TARGET_MIN,
    generate_approval_controlled_records,
)
from aiodoo_training.system_training_contract.generators.conversation_controlled import (
    CONVERSATION_CONTROLLED_GENERATOR,
    MIN_FAMILIES as CONVERSATION_MIN_FAMILIES,
    TARGET_MAX as CONVERSATION_TARGET_MAX,
    TARGET_MIN as CONVERSATION_TARGET_MIN,
    generate_conversation_controlled_records,
)
from aiodoo_training.system_training_contract.generators.evaluation_semantics import (
    EVALUATION_SEMANTIC_DEFINITION,
)
from aiodoo_training.system_training_contract.generators.mapping import (
    record_provider_capabilities,
)
from aiodoo_training.system_training_contract.generators.reasoning_sparse_emit import (
    BATCH2_CHECKSUM,
    analyze_approval_controlled,
    analyze_conversation_controlled,
    emit_reasoning_sparse_corpora,
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
SPARSE_TRAIN = WORKSPACE / "aiodoo-training" / "fixtures" / "fp2" / "reasoning_controlled_1"
SPARSE_DATA = WORKSPACE / "aiodoo-datasets" / "datasets" / "fp2" / "reasoning_controlled_1"


def test_conversation_mapping_and_types() -> None:
    records = generate_conversation_controlled_records()
    by_type = Counter(r["record_type"] for r in records)
    assert set(by_type) <= {"decision_context", "loop_decision"}
    assert by_type["decision_context"] >= 50
    assert by_type["loop_decision"] >= 50
    for r in records:
        assert r["provider_capability"] == "conversation"
        assert "conversation" in record_provider_capabilities(r["record_type"])
        validate_record_mapping(r)
        if r["record_type"] == "loop_decision":
            assert r["expected_output"]["decision_kind"] == "clarify"


def test_approval_mapping_and_kinds() -> None:
    records = generate_approval_controlled_records()
    assert {r["record_type"] for r in records} == {"loop_decision"}
    kinds = Counter(r["expected_output"]["decision_kind"] for r in records)
    assert set(kinds) <= {"approve", "reject", "modify"}
    assert kinds["approve"] >= 1 and kinds["reject"] >= 1 and kinds["modify"] >= 1
    for r in records:
        assert r["provider_capability"] == "approval"
        assert "approval" in record_provider_capabilities(r["record_type"])
        validate_record_mapping(r)


def test_evaluation_semantics_unresolved_no_generation() -> None:
    # AT-7.3 left Evaluation empty; AT-7.4 resolved mapping; AT-7.5 may emit corpus.
    # Conversation/Approval remain independent of Evaluation generation.
    assert EVALUATION_SEMANTIC_DEFINITION["record_type"] == "evaluation_judgment"
    man = json.loads((SPARSE_TRAIN / "evaluation" / "manifest.json").read_text(encoding="utf-8"))
    # After AT-7.5 the Evaluation tree holds a controlled corpus; still no legacy projection.
    assert man.get("legacy_projection") in {False, None} or man.get("legacy_projection_status") == "NOT PERFORMED"
    assert (SPARSE_TRAIN / "conversation" / "manifest.json").is_file()
    assert (SPARSE_TRAIN / "approval" / "manifest.json").is_file()


def test_generator_determinism() -> None:
    c1 = generate_conversation_controlled_records()
    c2 = generate_conversation_controlled_records()
    a1 = generate_approval_controlled_records()
    a2 = generate_approval_controlled_records()
    assert [json.dumps(r, sort_keys=True) for r in c1] == [
        json.dumps(r, sort_keys=True) for r in c2
    ]
    assert [json.dumps(r, sort_keys=True) for r in a1] == [
        json.dumps(r, sort_keys=True) for r in a2
    ]
    assert len({r["record_id"] for r in c1}) == len(c1)
    assert len({r["record_id"] for r in a1}) == len(a1)


def test_counts_families_splits_and_isolation() -> None:
    for gen, amin, amax, mfam in (
        (generate_conversation_controlled_records, CONVERSATION_TARGET_MIN, CONVERSATION_TARGET_MAX, CONVERSATION_MIN_FAMILIES),
        (generate_approval_controlled_records, APPROVAL_TARGET_MIN, APPROVAL_TARGET_MAX, APPROVAL_MIN_FAMILIES),
    ):
        records = gen()
        assert amin <= len(records) <= amax
        families = {r["metadata"]["scenario_family"] for r in records}
        assert len(families) >= mfam
        family_splits: dict[str, set[str]] = {}
        splits = Counter()
        for r in records:
            fam = r["metadata"]["scenario_family"]
            split = assign_split(r).value
            family_splits.setdefault(fam, set()).add(split)
            splits[split] += 1
        assert all(len(v) == 1 for v in family_splits.values())
        assert splits["validation"] > 0 and splits["test"] > 0


def test_duplicates_forbidden_how_and_legacy_exclusion() -> None:
    for records in (
        generate_conversation_controlled_records(),
        generate_approval_controlled_records(),
    ):
        assert find_duplicates(records)["duplicate_groups"] == 0
        assert all(not scan_forbidden_how(r) for r in records)
        assert all(r.get("metadata", {}).get("legacy") is False for r in records)
        assert all(
            not str(r["metadata"].get("quality_corpus") or "").startswith("negative")
            for r in records
        )
        blob = json.dumps(records)
        assert "conversation_dataset.jsonl" not in blob
        assert "approval_dataset.jsonl" not in blob
        assert "evaluation_dataset.jsonl" not in blob


def test_negative_controls_match_and_excluded() -> None:
    assert all(evaluate_negative_case(c)["matched"] for c in REASONING_SPARSE_NEGATIVE_CASES)
    # Mapping policy: planning_decision is never Conversation/Approval/Evaluation
    assert "conversation" not in record_provider_capabilities("planning_decision")
    assert "approval" not in record_provider_capabilities("planning_decision")
    assert "evaluation" not in record_provider_capabilities("planning_decision")
    assert "evaluation" not in record_provider_capabilities("loop_decision")


def test_analyze_ready_and_provider_dataset_equivalence() -> None:
    conv = analyze_conversation_controlled(generate_conversation_controlled_records())
    appr = analyze_approval_controlled(generate_approval_controlled_records())
    assert conv["verdict"] == "CONVERSATION_CORPUS_READY"
    assert appr["verdict"] == "APPROVAL_CORPUS_READY"
    assert conv["scorecard"]["provider_dataset_equivalence"] is True
    assert appr["scorecard"]["provider_dataset_equivalence"] is True
    assert conv["scorecard"]["family_leakage"] == 0
    assert appr["scorecard"]["family_leakage"] == 0
    assert conv["scorecard"]["negatives_ok"] is True
    assert appr["scorecard"]["negatives_ok"] is True


def test_emit_and_batch2_immutable(tmp_path: Path) -> None:
    before = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))["checksum"]
    assert before == BATCH2_CHECKSUM
    result = emit_reasoning_sparse_corpora(
        training_root=tmp_path / "train",
        datasets_root=tmp_path / "data",
    )
    assert result["version"] == REASONING_SPARSE_VERSION
    assert result["overall_verdict"] == "REASONING_SPARSE_DATA_PARTIAL"
    assert result["conversation"]["verdict"] == "CONVERSATION_CORPUS_READY"
    assert result["approval"]["verdict"] == "APPROVAL_CORPUS_READY"
    # Emit still writes Evaluation semantics tree without natives; contract status
    # may advance via evaluation_semantics module (AT-7.4) independently.
    assert result["evaluation"]["count"] == 0
    assert (tmp_path / "train" / "conversation" / "pack_reasoning.jsonl").is_file()
    assert (tmp_path / "train" / "approval" / "pack_reasoning.jsonl").is_file()
    assert (tmp_path / "train" / "evaluation" / "semantics_report.json").is_file()
    after = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))["checksum"]
    assert after == before == BATCH2_CHECKSUM
    assert not any("controlled_batch_2" in p for p in result["written"])


def test_on_disk_corpora_present() -> None:
    assert REASONING_SPARSE_VERSION == "fp2-reasoning-sparse-1.0.0"
    for root in (SPARSE_TRAIN, SPARSE_DATA):
        conv = json.loads((root / "conversation" / "manifest.json").read_text(encoding="utf-8"))
        appr = json.loads((root / "approval" / "manifest.json").read_text(encoding="utf-8"))
        ev = json.loads((root / "evaluation" / "manifest.json").read_text(encoding="utf-8"))
        assert conv["verdict"] == "CONVERSATION_CORPUS_READY"
        assert appr["verdict"] == "APPROVAL_CORPUS_READY"
        if ev.get("version") == "fp2-evaluation-controlled-1.0.0":
            assert ev["verdict"] == "EVALUATION_CORPUS_READY"
            assert ev["total_records"] == 252
        else:
            assert ev["total_records"] == 0
            assert ev["verdict"] == "EVALUATION_MAPPING_READY"
        assert CONVERSATION_TARGET_MIN <= conv["total_records"] <= CONVERSATION_TARGET_MAX
        assert APPROVAL_TARGET_MIN <= appr["total_records"] <= APPROVAL_TARGET_MAX
        assert conv["controlled_batch_2_modified"] is False
        assert appr["controlled_batch_2_modified"] is False
        assert conv["generator"] == CONVERSATION_CONTROLLED_GENERATOR
        assert appr["generator"] == APPROVAL_CONTROLLED_GENERATOR


def test_batch2_content_checksum_unchanged() -> None:
    import hashlib

    from aiodoo_training.system_training_contract.evaluation.harness import NATIVE_FAMILIES

    h = hashlib.sha256()
    for fam in NATIVE_FAMILIES:
        h.update((BATCH2 / f"{fam}.jsonl").read_bytes())
    assert h.hexdigest() == BATCH2_CHECKSUM
