"""Formatter tests for all protocol dataset types."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase1
from aiodoo_training.datasets.formatters.formatters import EvaluationFormatter
from aiodoo_training.datasets.reader import ProtocolRecordReader
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.exceptions import DomainError

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "datasets"

# Capabilities with aiodoo_contract projections render system/user/assistant
# via CapabilityPromptBuilder (ADR-0003). Fixture-backed cases exclude
# evaluation until Phase 3 refreshes the still-catalog-shaped fixture;
# evaluation is covered by dedicated judgment-record tests below.
CONTRACT_CASES = [
    ("coding", DatasetType.CODING),
    ("planner", DatasetType.PLANNER),
    ("repair", DatasetType.REPAIR),
    ("execution", DatasetType.EXECUTION),
    ("approval", DatasetType.APPROVAL),
    ("conversation", DatasetType.CONVERSATION),
]
NON_CONTRACT_CASES = [
    ("context", DatasetType.CONTEXT),
]

_JUDGMENT_RECORD = {
    "record_id": "EVL-test00000000000000000000000001",
    "candidate": {"capability": "coding", "output": {"goal": "x"}},
    "expectation": {"capability": "coding", "output": {"goal": "x"}},
    "rubric": "Judge coding quality",
    "verdict": "pass",
    "score": 1.0,
    "explanation": "matches expectation",
    "metadata": {"protocol_version": "1.0", "module": "sale"},
}


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase1(overwrite=True)


@pytest.mark.parametrize(("name", "dtype"), CONTRACT_CASES)
def test_contract_formatter_produces_system_user_assistant_messages(
    name: str, dtype: DatasetType
) -> None:
    source = JsonlDatasetSource(validate=False)
    ref = DatasetRef(path=FIXTURES / f"{name}.jsonl", dataset_type=dtype, protocol_version="1.0")
    examples = list(source.load([ref]))
    assert examples
    assert len(examples[0].messages) == 3
    assert examples[0].messages[0]["role"] == "system"
    assert examples[0].messages[1]["role"] == "user"
    assert examples[0].messages[2]["role"] == "assistant"
    assert examples[0].dataset_type == dtype
    assert examples[0].metadata["capability"] == dtype.value
    assert examples[0].metadata["contract_version"]


@pytest.mark.parametrize(("name", "dtype"), NON_CONTRACT_CASES)
def test_non_contract_formatter_produces_user_assistant_messages(
    name: str, dtype: DatasetType
) -> None:
    source = JsonlDatasetSource(validate=False)
    ref = DatasetRef(path=FIXTURES / f"{name}.jsonl", dataset_type=dtype, protocol_version="1.0")
    examples = list(source.load([ref]))
    assert examples
    assert len(examples[0].messages) == 2
    assert examples[0].messages[0]["role"] == "user"
    assert examples[0].messages[1]["role"] == "assistant"
    assert examples[0].dataset_type == dtype


def test_evaluation_formatter_uses_contract_pipeline() -> None:
    example = EvaluationFormatter().format(dict(_JUDGMENT_RECORD), "evaluation")
    assert example.dataset_type == DatasetType.EVALUATION
    assert len(example.messages) == 3
    assert example.messages[0]["role"] == "system"
    assert example.messages[1]["role"] == "user"
    assert example.messages[2]["role"] == "assistant"
    assert example.metadata["capability"] == "evaluation"
    assert example.metadata["contract_version"]

    label = json.loads(example.messages[2]["content"])
    assert label["capability"] == "evaluation"
    assert label["verdict"] == "pass"
    assert label["score"] == 1.0
    assert label["explanation"] == "matches expectation"
    assert "evaluation_id" not in label
    assert "catalog" not in example.messages[1]["content"]


def test_evaluation_formatter_rejects_catalog_shaped_record() -> None:
    catalog_record = {
        "evaluation_id": "EVALROOT-legacy",
        "catalog": {"catalog_id": "CTLG-1", "suites": []},
        "metadata": {"protocol_version": "1.0"},
    }
    with pytest.raises(DomainError, match="candidate"):
        EvaluationFormatter().format(catalog_record, "evaluation")


def test_reader_count_matches_fixture() -> None:
    assert ProtocolRecordReader().count_records(FIXTURES / "approval.jsonl") == 1
