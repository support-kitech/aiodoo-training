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
# via CapabilityPromptBuilder (ADR-0003). "context" keeps a user/assistant pair.
CONTRACT_CASES = [
    ("coding", DatasetType.CODING),
    ("planner", DatasetType.PLANNER),
    ("repair", DatasetType.REPAIR),
    ("execution", DatasetType.EXECUTION),
    ("approval", DatasetType.APPROVAL),
    ("conversation", DatasetType.CONVERSATION),
    ("evaluation", DatasetType.EVALUATION),
]
NON_CONTRACT_CASES = [
    ("context", DatasetType.CONTEXT),
]


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


def test_evaluation_formatter_label_is_evaluation_response() -> None:
    source = JsonlDatasetSource(validate=True)
    ref = DatasetRef(
        path=FIXTURES / "evaluation.jsonl",
        dataset_type=DatasetType.EVALUATION,
        protocol_version="1.0",
    )
    examples = list(source.load([ref]))
    assert examples
    label = json.loads(examples[0].messages[2]["content"])
    assert label["capability"] == "evaluation"
    assert label["verdict"] == "pass"
    assert "evaluation_id" not in label
    assert "catalog" not in examples[0].messages[1]["content"]
    assert examples[0].example_id == (
        "evaluation:EVL-fixture000000000000000000000001"
    )


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
