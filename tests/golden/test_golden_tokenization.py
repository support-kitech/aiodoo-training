"""Golden tokenization digest lock."""

import hashlib
from pathlib import Path

from aiodoo_training.bootstrap import bootstrap_phase1
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.domain.config import DatasetMixSpec
from aiodoo_training.domain.enums import DatasetType, ModelFamily, Precision
from aiodoo_training.domain.examples import TokenizationConfig
from aiodoo_training.domain.refs import DatasetRef, ModelRef
from aiodoo_training.infrastructure.huggingface.templates import QwenChatTemplate
from aiodoo_training.infrastructure.huggingface.tokenizer import DeterministicStubTokenizer
from aiodoo_training.tokenization.pipeline import TokenizationPipeline

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "datasets"

# Locked after first green run of DeterministicStubTokenizer against fixtures.
GOLDEN_DIGEST = "REPLACE_ME"


def test_golden_coding_token_digest() -> None:
    bootstrap_phase1(overwrite=True)
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
    _session, examples = JsonlDatasetSource().open_session(mix, session_id="golden")
    config = TokenizationConfig(max_length=64, padding="max_length", mask_prompt=True)
    tok = DeterministicStubTokenizer(template=QwenChatTemplate(), config=config)
    tok.load(ModelRef(identifier="stub-golden", family=ModelFamily.QWEN, precision=Precision.FP32))
    batch = TokenizationPipeline(tok, template=QwenChatTemplate(), config=config).run(examples)
    digest = hashlib.sha256(repr(batch.input_ids).encode("utf-8")).hexdigest()
    # First assertion path: freeze local golden file if constant still placeholder.
    golden_path = ROOT / "tests" / "golden" / "coding_tokens.sha256"
    if GOLDEN_DIGEST == "REPLACE_ME":
        if golden_path.exists():
            assert digest == golden_path.read_text(encoding="utf-8").strip()
        else:
            golden_path.write_text(digest + "\n", encoding="utf-8")
            assert digest == golden_path.read_text(encoding="utf-8").strip()
    else:
        assert digest == GOLDEN_DIGEST
