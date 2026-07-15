"""Golden: identical model + dataset + seed → identical evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase4
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.evaluation import EvaluationEngine, build_stub_evaluation_context


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase4(overwrite=True)


def _metric_sequence(report) -> list[tuple[str, float]]:
    return [(m.name, m.value) for m in report.metrics]


def test_golden_evaluation_determinism(tmp_path: Path) -> None:
    seed = 42
    dataset_path = Path("fixture/golden-eval.jsonl")
    dataset_ref = DatasetRef(
        path=dataset_path,
        dataset_type=DatasetType.EVALUATION,
        protocol_version="1.0",
        name="golden-eval",
    )

    ctx_a = build_stub_evaluation_context(
        output_dir=tmp_path / "run_a",
        seed=seed,
        dataset_path=dataset_path,
    )
    ctx_b = build_stub_evaluation_context(
        output_dir=tmp_path / "run_b",
        seed=seed,
        dataset_path=dataset_path,
    )

    engine = EvaluationEngine()
    _, report_a = engine.run(ctx_a)
    _, report_b = engine.run(ctx_b)

    assert _metric_sequence(report_a) == _metric_sequence(report_b)
    assert len(report_a.metrics) == 3
    assert {m.name for m in report_a.metrics} == {"loss", "perplexity", "token_accuracy"}
    _ = dataset_ref  # documents explicit dataset identity used via path
