"""AT-7.7 — Conversation path smoke: selection, config, immutability (no GPU retrain)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aiodoo_training.bootstrap import bootstrap_phase7
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.refs import RECORD_FORMAT_FP2_TRAINING_EXAMPLE, DatasetRef

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]
CORPUS = WORKSPACE / "aiodoo-datasets/datasets/fp2/reasoning_controlled_1/conversation"
SMOKE = WORKSPACE / "aiodoo-training/artifacts/at7_conversation/smoke"
CONFIG = WORKSPACE / "aiodoo-training/configs/training/at7_conversation_smoke/experiment.yaml"
CORPUS_CHECKSUM = "488b3a7576071c875c32e277c49562bb9c472904e32b12a1b98fcf6558da9de3"
BATCH2_CHECKSUM = "728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227"
BATCH2 = WORKSPACE / "aiodoo-datasets/datasets/fp2/controlled_batch_2"
APPR = WORKSPACE / "aiodoo-datasets/datasets/fp2/reasoning_controlled_1/approval/manifest.json"
EVAL = WORKSPACE / "aiodoo-datasets/datasets/fp2/reasoning_controlled_1/evaluation/manifest.json"
AUTHORIZED = frozenset({"decision_context", "loop_decision"})


def _corpus_checksum(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def test_conversation_pack_loads_fp2() -> None:
    bootstrap_phase7(overwrite=True)
    pack = CORPUS / "pack_reasoning.jsonl"
    ref = DatasetRef(
        path=pack,
        dataset_type=DatasetType.CONVERSATION,
        protocol_version="1.0.0",
        record_format=RECORD_FORMAT_FP2_TRAINING_EXAMPLE,
        name="conv_pack",
    )
    examples = list(JsonlDatasetSource(validate=True).load([ref]))
    assert len(examples) == 232
    assert all(ex.dataset_type == DatasetType.CONVERSATION for ex in examples)
    assert all(ex.metadata.get("provider_capability") == "conversation" for ex in examples)
    assert all(ex.metadata.get("record_type") in AUTHORIZED for ex in examples)
    assert all("AIODOO System Training Contract" in ex.messages[0]["content"] for ex in examples[:5])


def test_smoke_subset_family_isolation_and_semantics() -> None:
    man = json.loads((SMOKE / "smoke_manifest.json").read_text(encoding="utf-8"))
    assert man["smoke_train"] == 16
    assert man["smoke_validation"] == 4
    assert man["provider_capability"] == "conversation"
    assert man["product_plane"] == "reasoning"
    assert man["corpus_checksum"] == CORPUS_CHECKSUM
    assert man["native_count"] == 232
    assert man["approval_contamination"] == 0
    assert man["planner_contamination"] == 0
    assert man["evaluation_contamination"] == 0

    splits: dict[str, str] = {}
    for line in (CORPUS / "splits.jsonl").read_text().splitlines():
        row = json.loads(line)
        splits[row["record_id"]] = row["split"]

    for rid in man["train_record_ids"]:
        assert splits[rid] == "train"
    for rid in man["validation_record_ids"]:
        assert splits[rid] == "validation"
    assert set(man["train_record_ids"]).isdisjoint(man["validation_record_ids"])

    iso = man["family_isolation"]
    train_f = set(iso["train_families"])
    val_f = set(iso["validation_families"])
    test_f = set(iso["test_families"])
    assert train_f.isdisjoint(val_f)
    assert train_f.isdisjoint(test_f)
    assert val_f.isdisjoint(test_f)

    fps = [e["fingerprint"] for e in man["train"]["entries"] + man["validation"]["entries"]]
    assert len(fps) == len(set(fps))

    assert man["train"]["record_types"].get("decision_context") == 8
    assert man["train"]["record_types"].get("loop_decision") == 8
    assert man["train"]["decision_kinds"].get("clarify") == 8

    natives = {
        json.loads(line)["record_id"]: json.loads(line)
        for line in (CORPUS / "conversation_native.jsonl").read_text().splitlines()
        if line.strip()
    }
    for rid in man["train_record_ids"] + man["validation_record_ids"]:
        rec = natives[rid]
        assert rec["provider_capability"] == "conversation"
        assert rec["record_type"] in AUTHORIZED
        eo = rec.get("expected_output") or {}
        if rec["record_type"] == "loop_decision":
            assert eo.get("decision_kind") == "clarify"
            assert eo.get("decision_kind") not in {"approve", "reject", "modify"}
        assert "verdict" not in eo
        assert "score" not in eo or rec["record_type"] != "loop_decision"


def test_smoke_jsonl_all_conversation() -> None:
    for name in ("conversation_train_smoke.jsonl", "conversation_validation_smoke.jsonl"):
        for line in (SMOKE / "data" / name).read_text().splitlines():
            ex = json.loads(line)
            assert ex["dataset_type"] == "conversation"
            assert ex["metadata"]["provider_capability"] == "conversation"
            assert ex["metadata"]["record_type"] in AUTHORIZED
            assert {m["role"] for m in ex["messages"]} == {"user", "assistant"}


def test_config_is_smoke_scale() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert "max_steps: 4" in text
    assert "provider_capability: \"conversation\"" in text or "provider_capability: conversation" in text
    assert "product_plane: \"reasoning\"" in text or "product_plane: reasoning" in text
    assert "fp2_training_example" in text
    assert "pack_reasoning" in text
    assert CORPUS_CHECKSUM in text
    assert "Qwen2.5-Coder-3B-Instruct" in text
    assert "local_path:" in text


def test_smoke_result_pass() -> None:
    result = json.loads((SMOKE / "smoke_result.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "AT7_7_TRAINING_PATH_PASS"
    assert result["steps_executed"] == 4
    assert float(result["train_loss"]) == float(result["train_loss"])
    assert result["finite_logit_sanity"] is True
    assert result["adapter_reload"] is True
    assert result["adapter_weight_bytes"] > 0
    assert result["provider_capability"] == "conversation"
    assert result["product_plane"] == "reasoning"
    assert result["system_training_contract_version"] == "1.0.0"
    assert result["coding_adapter_loaded"] is False
    assert result["planner_adapter_loaded"] is False
    assert result["approval_adapter_loaded"] is False
    assert result["evaluation_adapter_loaded"] is False
    prov = json.loads((SMOKE / "fp2_provenance.json").read_text(encoding="utf-8"))
    assert prov["provider_capability"] == "conversation"
    assert prov["product_plane"] == "reasoning"
    assert prov["corpus_checksum"] == CORPUS_CHECKSUM


def test_immutability() -> None:
    assert _corpus_checksum(CORPUS) == CORPUS_CHECKSUM
    assert (SMOKE / "checksum_before.txt").read_text().strip() == CORPUS_CHECKSUM
    assert (SMOKE / "checksum_after.txt").read_text().strip() == CORPUS_CHECKSUM
    man = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))
    assert man["checksum"] == BATCH2_CHECKSUM
    assert json.loads(APPR.read_text())["total_records"] == 162
    ev = json.loads(EVAL.read_text())
    assert ev["total_records"] == 252
    assert ev["version"] == "fp2-evaluation-controlled-1.0.0"
