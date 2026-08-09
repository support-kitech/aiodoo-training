"""AT-7.4 — Evaluation FP2 Training Contract & mapping decision tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.system_training_contract.evaluation.harness import NATIVE_FAMILIES
from aiodoo_training.system_training_contract.generators.evaluation_semantics import (
    EVALUATION_CONTRACT_DECISION,
    EVALUATION_SEMANTIC_DEFINITION,
)
from aiodoo_training.system_training_contract.generators.mapping import (
    EVALUATION_ALLOWED_RECORD_TYPES,
    EVALUATION_REJECTED_RECORD_TYPES,
    assert_no_adapter_chain,
    record_provider_capabilities,
)
from aiodoo_training.system_training_contract.quality.formatters import (
    format_fp2_pack,
    format_fp2_record,
)
from aiodoo_training.system_training_contract.quality.negatives import (
    REASONING_SPARSE_NEGATIVE_CASES,
    evaluate_negative_case,
)
from aiodoo_training.system_training_contract.records import (
    EVALUATION_VERDICTS,
    RECORD_TYPES,
    EvaluationJudgmentRecord,
    TrainingRecordError,
    validate_record_mapping,
)

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]
BATCH2 = WORKSPACE / "aiodoo-datasets" / "datasets" / "fp2" / "controlled_batch_2"
BATCH2_CHECKSUM = "728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227"
SPARSE = WORKSPACE / "aiodoo-datasets" / "datasets" / "fp2" / "reasoning_controlled_1"


def _valid_judgment(**overrides: object) -> dict:
    base = {
        "training_contract_version": "1.0.0",
        "record_type": "evaluation_judgment",
        "record_id": "eval-test-001",
        "system_contract": "capability.evaluation",
        "provider_capability": "evaluation",
        "input": {
            "candidate": {"capability": "planner", "summary": "ship feature"},
            "expectation": {"capability": "planner", "summary": "ship feature"},
            "rubric": "Judge whether the planner candidate matches the expectation",
        },
        "expected_output": {
            "verdict": "pass",
            "score": 1.0,
            "explanation": "Candidate matches expectation",
        },
        "provenance": {},
        "metadata": {"scenario_family": "eval_test_family", "legacy": False},
    }
    base.update(overrides)
    return base


def test_contract_decision_status() -> None:
    assert EVALUATION_CONTRACT_DECISION == "EVALUATION_MAPPING_READY"
    assert EVALUATION_SEMANTIC_DEFINITION["status"] == "EVALUATION_MAPPING_READY"
    assert EVALUATION_SEMANTIC_DEFINITION["fp2_representation_resolved"] is True
    # AT-7.5 authorizes controlled generation after mapping readiness.
    assert EVALUATION_SEMANTIC_DEFINITION["fp2_generation_authorized"] is True
    assert EVALUATION_SEMANTIC_DEFINITION["legacy_projection"].startswith("NOT PERFORMED")
    assert EVALUATION_SEMANTIC_DEFINITION.get("fp2_corpus_verdict") == "EVALUATION_CORPUS_READY"


def test_record_type_registered() -> None:
    assert "evaluation_judgment" in RECORD_TYPES
    assert EVALUATION_VERDICTS == frozenset({"pass", "fail", "inconclusive"})


def test_system_aligned_request_response_shape() -> None:
    from aiodoo_contract.schemas.enums import EvaluationVerdict
    from aiodoo_contract.schemas.evaluation import EvaluationRequest, EvaluationResponse

    req = EvaluationRequest(
        candidate={"answer": 42},
        expectation={"answer": 42},
        rubric="match answers",
    )
    resp = EvaluationResponse(
        request_id=req.request_id,
        verdict=EvaluationVerdict.PASS,
        score=1.0,
        explanation="ok",
    )
    rec = EvaluationJudgmentRecord(
        record_type="evaluation_judgment",
        record_id="align-1",
        provider_capability="evaluation",
        candidate=req.candidate,
        expectation=req.expectation,
        rubric=req.rubric,
        verdict=resp.verdict.value,
        score=resp.score,
        explanation=resp.explanation,
    )
    dumped = rec.to_dict()
    assert dumped["input"]["candidate"] == {"answer": 42}
    assert dumped["expected_output"]["verdict"] == "pass"
    assert dumped["system_contract"] == "capability.evaluation"


def test_provider_mapping_allowed_and_rejected() -> None:
    assert EVALUATION_ALLOWED_RECORD_TYPES == frozenset({"evaluation_judgment"})
    assert record_provider_capabilities("evaluation_judgment") == frozenset({"evaluation"})
    for rtype in EVALUATION_REJECTED_RECORD_TYPES:
        assert "evaluation" not in record_provider_capabilities(rtype)
    assert_no_adapter_chain()


def test_verdict_and_score_validation() -> None:
    ok = validate_record_mapping(_valid_judgment())
    assert ok["expected_output"]["verdict"] == "pass"
    with pytest.raises(TrainingRecordError):
        validate_record_mapping(
            _valid_judgment(expected_output={"verdict": "maybe", "score": 0.5})
        )
    with pytest.raises(TrainingRecordError):
        validate_record_mapping(
            _valid_judgment(expected_output={"verdict": "pass", "score": 1.5})
        )
    with pytest.raises(TrainingRecordError):
        validate_record_mapping(_valid_judgment(input={"rubric": "x"}))
    with pytest.raises(TrainingRecordError):
        validate_record_mapping(
            _valid_judgment(provider_capability="planner")
        )


def test_inconclusive_and_optional_fields() -> None:
    row = _valid_judgment(
        input={"candidate": {"status": "unknown"}},
        expected_output={"verdict": "inconclusive"},
    )
    validated = validate_record_mapping(row)
    assert validated["expected_output"] == {"verdict": "inconclusive"}
    assert "score" not in validated["expected_output"]
    assert "expectation" not in validated["input"]


def test_pack_provider_dataset_equivalence() -> None:
    row = validate_record_mapping(_valid_judgment())
    ex = format_fp2_record(row)
    assert ex.dataset_type == DatasetType.EVALUATION
    pack = format_fp2_pack([row], pack="reasoning")
    assert len(pack) == 1
    assert pack[0].dataset_type.value == "evaluation"
    assert format_fp2_pack([row], pack="development") == []


def test_negative_controls() -> None:
    assert all(evaluate_negative_case(c)["matched"] for c in REASONING_SPARSE_NEGATIVE_CASES)
    ids = {c["case_id"] for c in REASONING_SPARSE_NEGATIVE_CASES}
    for required in (
        "neg_eval_missing_candidate",
        "neg_eval_invalid_verdict",
        "neg_eval_score_out_of_range",
        "neg_eval_forbidden_how",
        "neg_planner_as_evaluation_policy",
        "neg_approval_as_evaluation_policy",
        "neg_conversation_as_evaluation_policy",
        "neg_execution_as_evaluation_policy",
        "neg_observation_as_evaluation_policy",
        "neg_eval_meta_judge_stuffed_decision_context",
        "pos_eval_judgment_control",
    ):
        assert required in ids


def test_legacy_isolation_and_no_corpus_generation() -> None:
    # AT-7.4 authorized mapping but forbade generation; AT-7.5 may add corpus.
    # This test only asserts AT-7.4 mapping artifacts remain and legacy projection
    # was not performed by the contract decision module.
    assert EVALUATION_SEMANTIC_DEFINITION["legacy_projection"].startswith("NOT PERFORMED")
    assert EVALUATION_SEMANTIC_DEFINITION["record_type"] == "evaluation_judgment"
    conv = json.loads((SPARSE / "conversation" / "manifest.json").read_text(encoding="utf-8"))
    appr = json.loads((SPARSE / "approval" / "manifest.json").read_text(encoding="utf-8"))
    assert conv["total_records"] == 232
    assert appr["total_records"] == 162
    assert conv["version"] == appr["version"] == "fp2-reasoning-sparse-1.0.0"


def test_batch2_immutable() -> None:
    before = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))["checksum"]
    assert before == BATCH2_CHECKSUM
    h = hashlib.sha256()
    for fam in NATIVE_FAMILIES:
        h.update((BATCH2 / f"{fam}.jsonl").read_bytes())
    assert h.hexdigest() == BATCH2_CHECKSUM
    # New family is Training Contract additive — not part of batch_2 inventory.
    assert "evaluation_judgment" not in NATIVE_FAMILIES
    assert not (BATCH2 / "evaluation_judgment.jsonl").exists()
