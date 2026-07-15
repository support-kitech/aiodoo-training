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

CASES = [
    ("coding", DatasetType.CODING),
    ("planner", DatasetType.PLANNER),
    ("repair", DatasetType.REPAIR),
    ("context", DatasetType.CONTEXT),
    ("execution", DatasetType.EXECUTION),
    ("approval", DatasetType.APPROVAL),
    ("conversation", DatasetType.CONVERSATION),
    ("evaluation", DatasetType.EVALUATION),
]


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase1(overwrite=True)


@pytest.mark.parametrize(("name", "dtype"), CASES)
def test_formatter_produces_user_assistant_messages(name: str, dtype: DatasetType) -> None:
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
