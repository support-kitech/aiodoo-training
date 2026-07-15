#!/usr/bin/env python3
"""
Prepare datasets for training: validate, format, tokenize (Phase 1).

Does not train models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiodoo_training.bootstrap import bootstrap_phase1
from aiodoo_training.cli import run
from aiodoo_training.datasets.formatters import register_default_formatters
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.domain.config import DatasetMixSpec
from aiodoo_training.domain.enums import DatasetType, ModelFamily, Precision
from aiodoo_training.domain.examples import TokenizationConfig
from aiodoo_training.domain.refs import DatasetRef, ModelRef
from aiodoo_training.infrastructure.huggingface.templates import register_default_chat_templates
from aiodoo_training.infrastructure.huggingface.tokenizer import DeterministicStubTokenizer
from aiodoo_training.tokenization.pipeline import TokenizationPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and tokenize AIODOO datasets.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to JSONL dataset.")
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        choices=[t.value for t in DatasetType if t != DatasetType.MIXED],
    )
    parser.add_argument("--protocol-version", type=str, default="1.0")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--session-id", type=str, default="prepare-1")
    parser.add_argument("--limit", type=int, default=0, help="Optional example cap (0 = all).")
    args = parser.parse_args(argv)

    def _command() -> int:
        bootstrap_phase1()
        register_default_formatters()
        register_default_chat_templates()

        ref = DatasetRef(
            path=args.dataset,
            dataset_type=DatasetType(args.dataset_type),
            protocol_version=args.protocol_version,
        )
        mix = DatasetMixSpec(datasets=(ref,), shuffle=False, seed=42)
        source = JsonlDatasetSource()
        session, examples = source.open_session(mix, session_id=args.session_id)
        if args.limit > 0:
            examples = examples[: args.limit]

        config = TokenizationConfig(max_length=args.max_length, padding="max_length")
        tokenizer = DeterministicStubTokenizer(config=config)
        tokenizer.load(
            ModelRef(
                identifier="stub",
                family=ModelFamily.UNKNOWN,
                precision=Precision.FP32,
            )
        )
        pipeline = TokenizationPipeline(
            tokenizer,
            config=config,
            dataset_fingerprint=session.dataset_fingerprint or "",
        )
        batch = pipeline.run(examples)
        print(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "dataset_fingerprint": session.dataset_fingerprint,
                    "examples": len(examples),
                    "batch_width": len(batch.input_ids[0]) if batch.input_ids else 0,
                    "masked_tokens": sum(
                        1 for row in batch.labels for v in row if v == config.ignore_index
                    ),
                },
                indent=2,
            )
        )
        return 0

    return run(_command)


if __name__ == "__main__":
    raise SystemExit(main())
