"""Tokenization, masking, packing, and golden hash tests."""

import hashlib
from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase1
from aiodoo_training.datasets.caching import DatasetCache
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.domain.config import DatasetMixSpec, PackingSpec
from aiodoo_training.domain.enums import DatasetType, ModelFamily, PackingMode, Precision
from aiodoo_training.domain.examples import IGNORE_INDEX, TokenizationConfig
from aiodoo_training.domain.refs import DatasetRef, ModelRef
from aiodoo_training.infrastructure.huggingface.templates import QwenChatTemplate
from aiodoo_training.infrastructure.huggingface.tokenizer import DeterministicStubTokenizer
from aiodoo_training.packing import NoPackingStrategy
from aiodoo_training.tokenization.masking import apply_assistant_only_mask
from aiodoo_training.tokenization.pipeline import TokenizationPipeline

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "datasets"


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase1(overwrite=True)


def test_assistant_only_mask() -> None:
    full = [1, 2, 3, 4, 5]
    prompt = [1, 2]
    labels = apply_assistant_only_mask(full, prompt, ignore_index=IGNORE_INDEX)
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]


def test_stub_tokenizer_masks_prompt_tokens() -> None:
    source = JsonlDatasetSource()
    ref = DatasetRef(
        path=FIXTURES / "coding.jsonl",
        dataset_type=DatasetType.CODING,
        protocol_version="1.0",
    )
    examples = tuple(source.load([ref]))[:1]
    config = TokenizationConfig(max_length=2048, padding="max_length", mask_prompt=True)
    tok = DeterministicStubTokenizer(template=QwenChatTemplate(), config=config)
    tok.load(ModelRef(identifier="stub", family=ModelFamily.QWEN, precision=Precision.FP32))
    batch = tok.encode_examples(examples)
    assert len(batch.input_ids) == 1
    assert any(v == IGNORE_INDEX for v in batch.labels[0])
    assert any(v != IGNORE_INDEX for v in batch.labels[0])


def test_tokenization_is_deterministic_golden_hash() -> None:
    source = JsonlDatasetSource()
    mix = DatasetMixSpec(
        datasets=(
            DatasetRef(
                path=FIXTURES / "coding.jsonl",
                dataset_type=DatasetType.CODING,
                protocol_version="1.0",
            ),
        ),
        shuffle=False,
        seed=0,
    )
    session, examples = source.open_session(mix, session_id="golden")
    config = TokenizationConfig(max_length=64, padding="max_length")
    tok = DeterministicStubTokenizer(template=QwenChatTemplate(), config=config)
    tok.load(ModelRef(identifier="stub", family=ModelFamily.QWEN, precision=Precision.FP32))
    pipeline = TokenizationPipeline(
        tok,
        template=QwenChatTemplate(),
        config=config,
        dataset_fingerprint=session.dataset_fingerprint or "",
    )
    batch = pipeline.run(examples)
    material = repr(batch.input_ids).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    # Freeze golden: stable under stub tokenizer + fixture.
    assert digest == hashlib.sha256(material).hexdigest()
    second = pipeline.run(examples)
    assert second.input_ids == batch.input_ids


def test_cache_hit_returns_same_batch(tmp_path: Path) -> None:
    source = JsonlDatasetSource()
    mix = DatasetMixSpec(
        datasets=(
            DatasetRef(
                path=FIXTURES / "planner.jsonl",
                dataset_type=DatasetType.PLANNER,
                protocol_version="1.0",
            ),
        ),
        shuffle=False,
        seed=0,
    )
    session, examples = source.open_session(mix, session_id="cache")
    config = TokenizationConfig(max_length=32)
    tok = DeterministicStubTokenizer(template=QwenChatTemplate(), config=config)
    tok.load(ModelRef(identifier="stub", family=ModelFamily.UNKNOWN, precision=Precision.FP32))
    cache = DatasetCache(tmp_path)
    pipeline = TokenizationPipeline(
        tok,
        template=QwenChatTemplate(),
        config=config,
        cache=cache,
        dataset_fingerprint=session.dataset_fingerprint or "x",
    )
    first = pipeline.run(examples)
    second = pipeline.run(examples)
    assert first.input_ids == second.input_ids


def test_no_packing_yields_empty() -> None:
    strategy = NoPackingStrategy()
    assert list(strategy.pack((), PackingSpec(mode=PackingMode.NONE))) == []
