"""Production smoke: certified Evaluation v2 → validate → format → tokenize.

Uses the certified ``evaluation_dataset.jsonl`` when present (sibling
``aiodoo-datasets`` checkout). Skips if the corpus is not on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase1
from aiodoo_training.contract.adapters import project_evaluation
from aiodoo_training.datasets.formatters.formatters import EvaluationFormatter
from aiodoo_training.datasets.reader import ProtocolRecordReader
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.datasets.validation import DatasetValidator
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.examples import TokenizationConfig
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.exceptions import DomainError
from aiodoo_training.infrastructure.huggingface.templates import QwenChatTemplate
from aiodoo_training.infrastructure.huggingface.tokenizer import DeterministicStubTokenizer
from aiodoo_training.tokenization.pipeline import TokenizationPipeline

ROOT = Path(__file__).resolve().parents[2]
CERTIFIED = ROOT.parent / "aiodoo-datasets" / "datasets" / "evaluation_dataset.jsonl"
CATALOG = (
    ROOT.parent / "aiodoo-datasets" / "datasets" / "evaluation_benchmark_catalog.jsonl"
)

SMOKE_LIMIT = 8
_LEGACY_PREFIX = "Evaluate using the following catalog:"


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase1(overwrite=True)


@pytest.mark.skipif(
    not CERTIFIED.is_file(),
    reason="certified evaluation_dataset.jsonl not present",
)
def test_certified_evaluation_smoke_pipeline() -> None:
    ref = DatasetRef(
        path=CERTIFIED,
        dataset_type=DatasetType.EVALUATION,
        protocol_version="1.0",
    )
    DatasetValidator().validate_ref(ref, sample_limit=SMOKE_LIMIT)

    examples = []
    for index, example in enumerate(JsonlDatasetSource(validate=False).load([ref])):
        examples.append(example)
        if index + 1 >= SMOKE_LIMIT:
            break

    assert len(examples) == SMOKE_LIMIT
    for example in examples:
        assert len(example.messages) == 3
        assert [m["role"] for m in example.messages] == ["system", "user", "assistant"]
        assert example.metadata["capability"] == "evaluation"
        assert example.example_id.startswith("evaluation:EVL-")
        label = json.loads(example.messages[2]["content"])
        assert label["capability"] == "evaluation"
        assert label["verdict"] in {"pass", "fail", "inconclusive"}
        assert "evaluation_id" not in label
        assert not example.messages[1]["content"].startswith(_LEGACY_PREFIX)

    first = next(ProtocolRecordReader().iter_records(CERTIFIED))
    projection = project_evaluation(first)
    assert projection.capability == "evaluation"
    formatted = EvaluationFormatter().format(first, "evaluation")
    assert len(formatted.messages) == 3

    config = TokenizationConfig(max_length=512, mask_prompt=True)
    tok = DeterministicStubTokenizer(template=QwenChatTemplate(), config=config)
    batch = TokenizationPipeline(tok, template=QwenChatTemplate(), config=config).run(
        tuple(examples)
    )
    assert len(batch.example_ids) == SMOKE_LIMIT
    assert len(batch.input_ids) == SMOKE_LIMIT
    assert all(len(row) > 0 for row in batch.input_ids)


@pytest.mark.skipif(not CATALOG.is_file(), reason="benchmark catalog not present")
def test_certified_catalog_rejected_for_training() -> None:
    ref = DatasetRef(
        path=CATALOG,
        dataset_type=DatasetType.EVALUATION,
        protocol_version="1.0",
    )
    with pytest.raises(DomainError, match="BenchmarkCatalog"):
        DatasetValidator().validate_ref(ref, sample_limit=1)
