"""Dataset loading, mixing, validation, fingerprint, and cache tests."""

from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase1
from aiodoo_training.datasets.caching import DatasetCache, TokenCacheKey
from aiodoo_training.datasets.fingerprinting import (
    fingerprint_dataset_file,
    fingerprint_dataset_mix,
)
from aiodoo_training.datasets.reader import ProtocolRecordReader
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.datasets.validation import DatasetValidator
from aiodoo_training.domain.config import DatasetMixSpec
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.examples import IGNORE_INDEX, TokenBatch
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.exceptions import DomainError

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "datasets"


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase1(overwrite=True)


def _ref(name: str, dtype: DatasetType) -> DatasetRef:
    return DatasetRef(
        path=FIXTURES / f"{name}.jsonl",
        dataset_type=dtype,
        protocol_version="1.0",
    )


def test_protocol_record_reader_streams_jsonl() -> None:
    reader = ProtocolRecordReader()
    records = list(reader.iter_records(FIXTURES / "coding.jsonl"))
    assert len(records) == 2
    assert "instruction" in records[0]


def test_dataset_validator_accepts_coding_fixture() -> None:
    DatasetValidator().validate_ref(_ref("coding", DatasetType.CODING))


def test_dataset_validator_rejects_missing_file() -> None:
    bad = DatasetRef(
        path=FIXTURES / "missing.jsonl",
        dataset_type=DatasetType.CODING,
        protocol_version="1.0",
    )
    with pytest.raises(DomainError, match="does not exist"):
        DatasetValidator().validate_ref(bad)


def test_jsonl_source_formats_and_opens_session() -> None:
    source = JsonlDatasetSource()
    mix = DatasetMixSpec(datasets=(_ref("coding", DatasetType.CODING),), shuffle=False, seed=1)
    session, examples = source.open_session(mix, session_id="t1")
    assert session.dataset_fingerprint
    assert session.mix_fingerprint == session.dataset_fingerprint
    assert session.examples_total == len(examples)
    assert len(examples) == 2
    assert examples[0].messages[0]["role"] == "user"
    assert examples[0].messages[1]["role"] == "assistant"


def test_mix_is_deterministic() -> None:
    source = JsonlDatasetSource()
    mix = DatasetMixSpec(
        datasets=(
            _ref("coding", DatasetType.CODING),
            _ref("planner", DatasetType.PLANNER),
        ),
        shuffle=True,
        seed=99,
    )
    a = tuple(source.load_mix(mix))
    b = tuple(source.load_mix(mix))
    assert [x.example_id for x in a] == [x.example_id for x in b]


def test_dataset_fingerprint_stable() -> None:
    path = FIXTURES / "coding.jsonl"
    assert fingerprint_dataset_file(path) == fingerprint_dataset_file(path)
    refs = (_ref("coding", DatasetType.CODING),)
    assert fingerprint_dataset_mix(refs, shuffle=False, seed=0) == fingerprint_dataset_mix(
        refs, shuffle=False, seed=0
    )


def test_token_cache_roundtrip(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    key = TokenCacheKey("d", "t", "c", "m", "cfg")
    batch = TokenBatch(
        example_ids=("a",),
        input_ids=((1, 2, 0),),
        attention_mask=((1, 1, 0),),
        labels=((IGNORE_INDEX, 2, IGNORE_INDEX),),
    )
    cache.put(key, batch)
    loaded = cache.get(key)
    assert loaded is not None
    assert loaded.input_ids == batch.input_ids
    # Key change invalidates
    other = TokenCacheKey("d2", "t", "c", "m", "cfg")
    assert cache.get(other) is None
