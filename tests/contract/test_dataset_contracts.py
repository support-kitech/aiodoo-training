"""Contract tests against aiodoo-datasets protocol fixtures."""

from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase1
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.datasets.validation import DatasetValidator
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.refs import DatasetRef

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "datasets"

CONTRACTS = [
    ("coding", DatasetType.CODING, {"instruction", "output", "metadata"}),
    ("planner", DatasetType.PLANNER, {"instruction", "output", "metadata"}),
    ("repair", DatasetType.REPAIR, {"instruction", "output", "metadata"}),
    ("context", DatasetType.CONTEXT, {"id", "query", "metadata"}),
    ("execution", DatasetType.EXECUTION, {"instruction", "output", "metadata"}),
    ("approval", DatasetType.APPROVAL, {"review_id", "decision", "metadata"}),
    ("conversation", DatasetType.CONVERSATION, {"instruction", "output", "metadata"}),
    ("evaluation", DatasetType.EVALUATION, {"evaluation_id", "metadata"}),
]


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase1(overwrite=True)


@pytest.mark.parametrize(("name", "dtype", "required"), CONTRACTS)
def test_protocol_contract(name: str, dtype: DatasetType, required: set[str]) -> None:
    ref = DatasetRef(
        path=FIXTURES / f"{name}.jsonl",
        dataset_type=dtype,
        protocol_version="1.0",
    )
    DatasetValidator().validate_ref(ref)
    examples = list(JsonlDatasetSource().load([ref]))
    assert examples
    assert examples[0].dataset_type == dtype
    # Formatter must always emit user/assistant pair for Phase 1 SFT.
    assert {m["role"] for m in examples[0].messages} == {"user", "assistant"}
