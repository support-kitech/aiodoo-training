"""AT-6.3 — Context path smoke: filter, subset, config, immutability (no GPU train)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aiodoo_training.bootstrap import bootstrap_phase7
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.refs import RECORD_FORMAT_FP2_TRAINING_EXAMPLE, DatasetRef

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]
CORPUS = WORKSPACE / "aiodoo-datasets/datasets/fp2/context_controlled_1"
SMOKE = WORKSPACE / "aiodoo-training/artifacts/at6_context/smoke"
CONFIG = WORKSPACE / "aiodoo-training/configs/training/at6_context_smoke/experiment.yaml"
BATCH2 = WORKSPACE / "aiodoo-datasets/datasets/fp2/controlled_batch_2"
AT62 = WORKSPACE / "aiodoo-training/fixtures/fp2/context"
LEGACY = WORKSPACE / "aiodoo-datasets/datasets/context_v1_0.jsonl"
BATCH2_CHECKSUM = "728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227"
CORPUS_CHECKSUM = "78e3b464d51b7e15912ca9aabc4ce65a579c4fbaddf5fa917609ac63c50ead87"


def _corpus_checksum() -> str:
    h = hashlib.sha256()
    for name in sorted(
        [
            "context_native.jsonl",
            "capability_intent.jsonl",
            "observation.jsonl",
            "pack_context.jsonl",
            "splits.jsonl",
            "manifest.json",
            "quality_report.json",
            "generation_metadata.json",
        ]
    ):
        p = CORPUS / name
        if p.is_file():
            h.update(name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_context_pack_provider_filter() -> None:
    bootstrap_phase7(overwrite=True)
    pack = CORPUS / "pack_context.jsonl"
    ref = DatasetRef(
        path=pack,
        dataset_type=DatasetType.CONTEXT,
        protocol_version="1.0.0",
        record_format=RECORD_FORMAT_FP2_TRAINING_EXAMPLE,
        name="ctx_pack",
    )
    examples = list(JsonlDatasetSource(validate=True).load([ref]))
    assert len(examples) == 261
    for ex in examples:
        assert ex.dataset_type == DatasetType.CONTEXT
        assert ex.metadata.get("provider_capability") == "context"
        assert "AIODOO System Training Contract" in ex.messages[0]["content"]


def test_smoke_subset_deterministic_and_strict() -> None:
    man = json.loads((SMOKE / "smoke_manifest.json").read_text(encoding="utf-8"))
    assert man["smoke_train"] == 16
    assert man["smoke_validation"] == 4
    assert man["provider_capability"] == "context"
    assert man["corpus_checksum"] == CORPUS_CHECKSUM
    splits: dict[str, str] = {}
    for line in (CORPUS / "splits.jsonl").read_text().splitlines():
        row = json.loads(line)
        splits[row["record_id"]] = row["split"]
    for rid in man["train_record_ids"]:
        assert splits[rid] == "train"
    for rid in man["validation_record_ids"]:
        assert splits[rid] == "validation"
    assert set(man["train_record_ids"]).isdisjoint(man["validation_record_ids"])
    assert man["train"]["record_types"].get("capability_intent") == 8
    assert man["train"]["record_types"].get("observation") == 8
    for cap in (
        "workspace.search",
        "workspace.navigate",
        "workspace.read",
        "repository.inspect",
    ):
        assert man["train"]["capabilities"].get(cap) == 4


def test_smoke_jsonl_all_context() -> None:
    for name in ("context_train_smoke.jsonl", "context_validation_smoke.jsonl"):
        for line in (SMOKE / name).read_text().splitlines():
            ex = json.loads(line)
            assert ex["dataset_type"] == "context"
            assert ex["metadata"]["provider_capability"] == "context"
            assert {m["role"] for m in ex["messages"]} == {"user", "assistant"}


def test_config_is_smoke_scale() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert "max_steps: 4" in text
    assert "provider_capability: \"context\"" in text or "provider_capability: context" in text
    assert "fp2-context-controlled-1.0.0" in text
    assert CORPUS_CHECKSUM in text
    assert "local_path:" in text
    assert "Qwen2.5-Coder-3B-Instruct" in text


def test_train_result_pass() -> None:
    result = json.loads((SMOKE / "train_result.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "AT6_3_TRAINING_PATH_PASS"
    assert result["steps_executed"] == 4
    assert result["train_loss"] is not None and float(result["train_loss"]) == float(
        result["train_loss"]
    )
    assert result["finite_logit_sanity"] is True
    assert result["adapter_reload"] is True
    assert result["adapter_weight_bytes"] > 0
    assert result["coding_adapter_loaded"] is False
    assert result["repair_adapter_loaded"] is False
    assert result["system_training_contract_version"] == "1.0.0"
    prov_path = SMOKE / "export"
    fp2 = next(prov_path.rglob("fp2_provenance.json"))
    prov = json.loads(fp2.read_text(encoding="utf-8"))
    assert prov["system_training_contract_version"] == "1.0.0"
    assert prov["provider_contract_version"] == "1.0.0"
    assert prov["provider_capability"] == "context"
    assert prov["corpus_checksum"] == CORPUS_CHECKSUM


def test_immutability() -> None:
    assert _corpus_checksum() == CORPUS_CHECKSUM
    assert (SMOKE / "checksum_before.txt").read_text().strip() == CORPUS_CHECKSUM
    assert (SMOKE / "checksum_after.txt").read_text().strip() == CORPUS_CHECKSUM
    b2 = json.loads((BATCH2 / "manifest.json").read_text())["checksum"]
    assert b2 == BATCH2_CHECKSUM
    at62 = json.loads((AT62 / "manifest.json").read_text())
    assert at62["version"] == "fp2-context-1.0.0"
    assert at62["total_records"] == 26
    assert LEGACY.is_file()
    # do not rewrite controlled corpus files from smoke outputs
    assert not (CORPUS / "context_train_smoke.jsonl").exists()
