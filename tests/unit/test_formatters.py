"""Formatter tests for all protocol dataset types."""

from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase1
from aiodoo_training.datasets.reader import ProtocolRecordReader
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.refs import DatasetRef

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "datasets"

# The six capabilities aiodoo_contract defines a shape for render a default
# system prompt via CapabilityPromptBuilder (ADR-0003), so their examples
# carry 3 messages (system, user, assistant). "context" and "evaluation"
# have no contract projection (see formatters.py module docstring) and keep
# the prior 2-message (user, assistant) shape.
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
    ("evaluation", DatasetType.EVALUATION),
]
CASES = CONTRACT_CASES + NON_CONTRACT_CASES


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


def test_reader_count_matches_fixture() -> None:
    assert ProtocolRecordReader().count_records(FIXTURES / "approval.jsonl") == 1
