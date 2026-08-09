"""AT-7.2 — Planner path smoke: selection, config, immutability (no GPU retrain)."""

from __future__ import annotations

import json
from pathlib import Path

from aiodoo_training.bootstrap import bootstrap_phase7
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.refs import RECORD_FORMAT_FP2_TRAINING_EXAMPLE, DatasetRef

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]
BATCH2 = WORKSPACE / "aiodoo-datasets/datasets/fp2/controlled_batch_2"
SMOKE = WORKSPACE / "aiodoo-training/artifacts/at7_planner/smoke"
CONFIG = WORKSPACE / "aiodoo-training/configs/training/at7_planner_smoke/experiment.yaml"
BATCH2_CHECKSUM = "728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227"


def test_reasoning_planner_pack_a_equals_b() -> None:
    bootstrap_phase7(overwrite=True)
    pack = BATCH2 / "pack_reasoning.jsonl"
    ref = DatasetRef(
        path=pack,
        dataset_type=DatasetType.PLANNER,
        protocol_version="1.0.0",
        record_format=RECORD_FORMAT_FP2_TRAINING_EXAMPLE,
        name="rea_pack",
    )
    examples = list(JsonlDatasetSource(validate=True).load([ref]))
    planner = [
        ex
        for ex in examples
        if ex.dataset_type == DatasetType.PLANNER
        and ex.metadata.get("provider_capability") == "planner"
    ]
    assert len(planner) == 580
    assert all(ex.dataset_type.value == "planner" for ex in planner)
    assert all(ex.metadata.get("provider_capability") == "planner" for ex in planner)
    assert all("AIODOO System Training Contract" in ex.messages[0]["content"] for ex in planner[:5])


def test_smoke_subset_strict_and_diverse() -> None:
    man = json.loads((SMOKE / "smoke_manifest.json").read_text(encoding="utf-8"))
    assert man["smoke_train"] == 16
    assert man["smoke_validation"] == 4
    assert man["provider_capability"] == "planner"
    assert man["product_plane"] == "reasoning"
    assert man["corpus_checksum"] == BATCH2_CHECKSUM
    assert man["reasoning_pack_planner_count"] == 580

    splits: dict[str, str] = {}
    for line in (BATCH2 / "splits.jsonl").read_text().splitlines():
        row = json.loads(line)
        splits[row["record_id"]] = row["split"]

    for rid in man["train_record_ids"]:
        assert splits[rid] == "train"
    for rid in man["validation_record_ids"]:
        assert splits[rid] == "validation"
    assert set(man["train_record_ids"]).isdisjoint(man["validation_record_ids"])

    types = man["train"]["record_types"]
    assert types.get("planning_decision") == 4
    assert types.get("engineering_feedback") == 4
    assert types.get("decision_context") == 4
    assert types.get("loop_decision") == 4
    # engineering_state is Development-only — not in Reasoning pack
    assert "engineering_state" not in types

    fps = [e["fingerprint"] for e in man["train"]["entries"] + man["validation"]["entries"]]
    assert len(fps) == len(set(fps))


def test_smoke_jsonl_all_planner() -> None:
    for name in ("planner_train_smoke.jsonl", "planner_validation_smoke.jsonl"):
        for line in (SMOKE / "data" / name).read_text().splitlines():
            ex = json.loads(line)
            assert ex["dataset_type"] == "planner"
            assert ex["metadata"]["provider_capability"] == "planner"
            assert {m["role"] for m in ex["messages"]} == {"user", "assistant"}


def test_config_is_smoke_scale() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert "max_steps: 4" in text
    assert "provider_capability: \"planner\"" in text or "provider_capability: planner" in text
    assert "product_plane: \"reasoning\"" in text or "product_plane: reasoning" in text
    assert "fp2_training_example" in text
    assert "pack_reasoning" in text
    assert BATCH2_CHECKSUM in text
    assert "Qwen2.5-Coder-3B-Instruct" in text
    assert "local_path:" in text


def test_smoke_result_pass() -> None:
    result = json.loads((SMOKE / "smoke_result.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "AT7_2_TRAINING_PATH_PASS"
    assert result["steps_executed"] == 4
    assert float(result["train_loss"]) == float(result["train_loss"])
    assert result["finite_logit_sanity"] is True
    assert result["adapter_reload"] is True
    assert result["adapter_weight_bytes"] > 0
    assert result["provider_capability"] == "planner"
    assert result["product_plane"] == "reasoning"
    assert result["system_training_contract_version"] == "1.0.0"
    assert result["coding_adapter_loaded"] is False
    prov = json.loads((SMOKE / "fp2_provenance.json").read_text(encoding="utf-8"))
    assert prov["provider_capability"] == "planner"
    assert prov["product_plane"] == "reasoning"
    assert prov["corpus_checksum"] == BATCH2_CHECKSUM


def test_batch2_immutable() -> None:
    man = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))
    assert man["checksum"] == BATCH2_CHECKSUM
    assert (SMOKE / "checksum_before.txt").read_text().strip() == BATCH2_CHECKSUM
    assert (SMOKE / "checksum_after.txt").read_text().strip() == BATCH2_CHECKSUM
