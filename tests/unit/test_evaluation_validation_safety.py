"""DatasetValidator safety + Evaluation v2 judgment schema tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase1
from aiodoo_training.datasets.mixing import stable_example_id
from aiodoo_training.datasets.validation import DatasetValidator
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.exceptions import DomainError

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "datasets"


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase1(overwrite=True)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _judgment(**overrides: object) -> dict:
    row = {
        "record_id": "EVL-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "candidate_id": "CAND-bbbbbbbbbbbbbbbbbbbbbbbb",
        "evaluation_case_key": "pass",
        "capability_under_test": "coding",
        "candidate": {"capability": "coding", "output": {"goal": "x"}},
        "expectation": {"capability": "coding", "output": {"goal": "x"}},
        "rubric": "Judge coding",
        "verdict": "pass",
        "score": 1.0,
        "explanation": "ok",
        "metadata": {
            "protocol_version": "1.0",
            "schema_version": "2.0",
            "module": "sale",
        },
    }
    row.update(overrides)
    return row


def test_validator_accepts_evaluation_fixture() -> None:
    ref = DatasetRef(
        path=FIXTURES / "evaluation.jsonl",
        dataset_type=DatasetType.EVALUATION,
        protocol_version="1.0",
    )
    DatasetValidator().validate_ref(ref)


def test_validator_rejects_benchmark_catalog_filename(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_benchmark_catalog.jsonl"
    _write_jsonl(path, [_judgment()])
    ref = DatasetRef(path=path, dataset_type=DatasetType.EVALUATION, protocol_version="1.0")
    with pytest.raises(DomainError, match="BenchmarkCatalog"):
        DatasetValidator().validate_ref(ref)


def test_validator_rejects_catalog_shaped_record(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_dataset.jsonl"
    _write_jsonl(
        path,
        [
            {
                "evaluation_id": "EVALROOT-1",
                "catalog": {"catalog_id": "CTLG-1", "suites": []},
                "metadata": {"protocol_version": "1.0"},
            }
        ],
    )
    ref = DatasetRef(path=path, dataset_type=DatasetType.EVALUATION, protocol_version="1.0")
    with pytest.raises(DomainError, match="BenchmarkCatalog"):
        DatasetValidator().validate_ref(ref)


def test_validator_rejects_training_forbidden(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_dataset.jsonl"
    _write_jsonl(
        path,
        [
            _judgment(
                metadata={
                    "protocol_version": "1.0",
                    "schema_version": "2.0",
                    "training_forbidden": True,
                }
            )
        ],
    )
    ref = DatasetRef(path=path, dataset_type=DatasetType.EVALUATION, protocol_version="1.0")
    with pytest.raises(DomainError, match="training_forbidden"):
        DatasetValidator().validate_ref(ref)


def test_validator_rejects_legacy_evaluation_id_only(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_dataset.jsonl"
    _write_jsonl(
        path,
        [
            {
                "evaluation_id": "EVALROOT-legacy",
                "metadata": {"protocol_version": "1.0"},
            }
        ],
    )
    ref = DatasetRef(path=path, dataset_type=DatasetType.EVALUATION, protocol_version="1.0")
    with pytest.raises(DomainError, match="missing required fields"):
        DatasetValidator().validate_ref(ref)


def test_validator_rejects_benchmark_catalog_manifest_name(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_dataset.jsonl"
    _write_jsonl(path, [_judgment()])
    manifest = tmp_path / "evaluation_dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_name": "benchmark_catalog",
                "protocol_version": "1.0",
            }
        ),
        encoding="utf-8",
    )
    ref = DatasetRef(path=path, dataset_type=DatasetType.EVALUATION, protocol_version="1.0")
    with pytest.raises(DomainError, match="BenchmarkCatalog"):
        DatasetValidator().validate_ref(ref)


def test_stable_example_id_prefers_record_id() -> None:
    record = {
        "record_id": "EVL-abc",
        "evaluation_id": "EVALROOT-should-not-win",
        "id": "should-not-win",
    }
    assert stable_example_id("evaluation", record, 0) == "evaluation:EVL-abc"
