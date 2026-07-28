"""Contract tests against aiodoo-datasets protocol fixtures."""

from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase1
from aiodoo_training.datasets.reader import ProtocolRecordReader
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.datasets.validation import DatasetValidator
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.refs import DatasetRef

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "datasets"

# Dataset types with a canonical aiodoo_contract projection render a
# default system prompt via CapabilityPromptBuilder (ADR-0003) and so emit
# a system/user/assistant triple. "context" has no contract projection
# (not a capability). EvaluationFormatter is contract-backed (Phase 2) but
# the on-disk evaluation fixture remains catalog-shaped until Phase 3 —
# structural required-field checks only for that fixture row.
CONTRACTS = [
    ("coding", DatasetType.CODING, {"instruction", "output", "metadata"}, True),
    ("planner", DatasetType.PLANNER, {"instruction", "output", "metadata"}, True),
    ("repair", DatasetType.REPAIR, {"instruction", "output", "metadata"}, True),
    ("context", DatasetType.CONTEXT, {"id", "query", "metadata"}, False),
    ("execution", DatasetType.EXECUTION, {"instruction", "output", "metadata"}, True),
    ("approval", DatasetType.APPROVAL, {"review_id", "decision", "metadata"}, True),
    ("conversation", DatasetType.CONVERSATION, {"instruction", "output", "metadata"}, True),
    ("evaluation", DatasetType.EVALUATION, {"evaluation_id", "metadata"}, False),
]


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase1(overwrite=True)


@pytest.mark.parametrize(("name", "dtype", "required", "has_contract_projection"), CONTRACTS)
def test_protocol_contract(
    name: str, dtype: DatasetType, required: set[str], has_contract_projection: bool
) -> None:
    ref = DatasetRef(
        path=FIXTURES / f"{name}.jsonl",
        dataset_type=dtype,
        protocol_version="1.0",
    )
    if dtype == DatasetType.EVALUATION:
        # Catalog-shaped fixture cannot be formatted or contract-validated
        # until Phase 3 refreshes fixtures / REQUIRED_FIELDS.
        record = next(ProtocolRecordReader().iter_records(ref.path))
        missing = required - record.keys()
        assert not missing, missing
        return

    DatasetValidator().validate_ref(ref)
    examples = list(JsonlDatasetSource().load([ref]))
    assert examples
    assert examples[0].dataset_type == dtype
    expected_roles = (
        {"system", "user", "assistant"} if has_contract_projection else {"user", "assistant"}
    )
    assert {m["role"] for m in examples[0].messages} == expected_roles
