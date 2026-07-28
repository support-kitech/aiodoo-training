"""Unit tests for Evaluation contract projection.

Covers ``project_evaluation`` registration and request/response mapping.
"""

from __future__ import annotations

import pytest
from aiodoo_contract.schemas.enums import EvaluationVerdict
from aiodoo_contract.schemas.evaluation import EvaluationRequest, EvaluationResponse
from aiodoo_contract.validators import ContractValidator

from aiodoo_training.contract.adapters import (
    SUPPORTED_CAPABILITIES,
    ContractAdapterError,
    project_evaluation,
    project_record,
)

_validator = ContractValidator()


def _assert_valid(projection) -> None:  # type: ignore[no-untyped-def]
    request_result = _validator.validate_request(projection.request)
    response_result = _validator.validate_response(projection.response)
    assert request_result.valid, request_result.issues
    assert response_result.valid, response_result.issues


class TestSupportedCapabilities:
    def test_evaluation_is_registered(self) -> None:
        assert "evaluation" in SUPPORTED_CAPABILITIES
        assert set(SUPPORTED_CAPABILITIES) == {
            "planner",
            "coding",
            "repair",
            "execution",
            "conversation",
            "approval",
            "evaluation",
        }


class TestProjectEvaluation:
    def test_pass_judgment_projects_and_validates(self) -> None:
        record = {
            "candidate": {"capability": "coding", "output": {"goal": "x"}},
            "expectation": {"capability": "coding", "output": {"goal": "x"}},
            "rubric": "Judge coding",
            "verdict": "pass",
            "score": 1.0,
            "explanation": "ok",
        }
        projection = project_evaluation(record)
        assert projection.capability == "evaluation"
        assert isinstance(projection.request, EvaluationRequest)
        assert isinstance(projection.response, EvaluationResponse)
        assert projection.response.verdict == EvaluationVerdict.PASS
        assert projection.response.score == 1.0
        assert projection.response.explanation == "ok"
        assert projection.request.candidate["capability"] == "coding"
        assert projection.request.rubric == "Judge coding"
        assert projection.response.request_id == projection.request.request_id
        _assert_valid(projection)

    def test_fail_judgment_projects(self) -> None:
        record = {
            "candidate": {"capability": "approval", "output": {}},
            "expectation": {"capability": "approval"},
            "verdict": "FAIL",
            "score": 0.0,
            "explanation": "missing structure",
        }
        projection = project_evaluation(record)
        assert projection.response.verdict == EvaluationVerdict.FAIL
        _assert_valid(projection)

    def test_inconclusive_allows_null_expectation(self) -> None:
        record = {
            "candidate": {"capability": "planner", "output": {"goal": "y"}},
            "expectation": None,
            "verdict": "inconclusive",
            "score": None,
        }
        projection = project_evaluation(record)
        assert projection.response.verdict == EvaluationVerdict.INCONCLUSIVE
        assert projection.request.expectation is None
        assert projection.response.score is None
        _assert_valid(projection)

    def test_project_record_dispatch(self) -> None:
        record = {
            "candidate": {"capability": "coding", "output": {"goal": "z"}},
            "verdict": "pass",
            "score": 1.0,
        }
        projection = project_record("evaluation", record)
        assert projection.capability == "evaluation"
        _assert_valid(projection)

    def test_missing_candidate_raises(self) -> None:
        with pytest.raises(ContractAdapterError, match="candidate"):
            project_evaluation({"verdict": "pass"})

    def test_invalid_expectation_type_raises(self) -> None:
        with pytest.raises(ContractAdapterError, match="expectation"):
            project_evaluation(
                {
                    "candidate": {"capability": "coding"},
                    "expectation": "not-a-mapping",
                    "verdict": "pass",
                }
            )

    def test_unmappable_verdict_raises(self) -> None:
        with pytest.raises(ContractAdapterError, match="verdict"):
            project_evaluation(
                {
                    "candidate": {"capability": "coding"},
                    "verdict": "maybe",
                }
            )

    def test_score_out_of_range_raises(self) -> None:
        with pytest.raises(ContractAdapterError, match="score"):
            project_evaluation(
                {
                    "candidate": {"capability": "coding"},
                    "verdict": "pass",
                    "score": 1.5,
                }
            )

    def test_invalid_score_type_raises(self) -> None:
        with pytest.raises(ContractAdapterError, match="score"):
            project_evaluation(
                {
                    "candidate": {"capability": "coding"},
                    "verdict": "pass",
                    "score": "high",
                }
            )
