"""Capability Package contract tests (Phase B1 / Option A)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiodoo_contract.version import CONTRACT_VERSION

from aiodoo_training.artifacts.output_layout import ArtifactOutputLayout
from aiodoo_training.artifacts.output_manager import ArtifactOutputManager
from aiodoo_training.artifacts.publish_contract import (
    ADAPTER_PROTOCOL_ARTIFACT_TYPE,
    ARTIFACT_METADATA_FILENAME,
    DEFAULT_SUPPORTED_ODOO_VERSIONS,
    MERGED_PROTOCOL_ARTIFACT_TYPE,
    build_adapter_artifact_json,
    build_base_model_artifact_json,
    build_merged_artifact_json,
    infer_adapter_artifact_type,
    resolve_capability_id,
)
from aiodoo_training.naming import TRAINING_IDS


def _base_resolved(capability: str, **extra: object) -> dict:
    payload: dict = {
        "name": capability,
        "experiment": {"id": capability, "stage": capability},
        "datasets": [{"path": f"{capability}.jsonl", "dataset_type": capability}],
        "dataset_version": "v1.0.0",
        "model": {"base_model": "Qwen/Qwen3-8B", "family": "qwen"},
        "adaptation": {"adapter_type": "qlora", "strategy": "qlora"},
    }
    payload.update(extra)
    return payload


@pytest.mark.parametrize("capability", list(TRAINING_IDS))
def test_resolve_capability_id_for_catalog(capability: str) -> None:
    assert resolve_capability_id(_base_resolved(capability)) == capability


@pytest.mark.parametrize("capability", list(TRAINING_IDS))
def test_protocol_artifact_type_always_coding_adapter(capability: str) -> None:
    assert infer_adapter_artifact_type(_base_resolved(capability)) == ADAPTER_PROTOCOL_ARTIFACT_TYPE


@pytest.mark.parametrize("capability", list(TRAINING_IDS))
def test_adapter_artifact_json_self_describing(capability: str) -> None:
    payload = build_adapter_artifact_json(
        experiment_id=f"aiodoo-{capability}",
        resolved=_base_resolved(capability),
        source_checkpoint="/tmp/checkpoint-1",
        created_at="2026-07-19T12:00:00Z",
    )
    assert payload["artifact_type"] == ADAPTER_PROTOCOL_ARTIFACT_TYPE
    assert payload["capability_id"] == capability
    assert payload["adapter_type"] == capability
    assert payload["peft_type"] == "qlora"
    assert payload["protocol_major"] == 1
    assert payload["identifier"] == f"aiodoo-{capability}"
    assert payload["model_family"] == "qwen"
    assert payload["architecture"] == "qwen"
    assert payload["supported_odoo_versions"] == list(DEFAULT_SUPPORTED_ODOO_VERSIONS)
    assert payload["dataset_version"] == "v1.0.0"
    assert payload["training_source"] == capability
    assert payload["producer"] == "aiodoo-training"
    assert payload["training_version"]
    assert payload["created_at"] == "2026-07-19T12:00:00Z"
    assert payload["source_checkpoint"] == "/tmp/checkpoint-1"


@pytest.mark.parametrize("capability", list(TRAINING_IDS))
def test_adapter_artifact_json_carries_contract_version(capability: str) -> None:
    payload = build_adapter_artifact_json(
        experiment_id=f"aiodoo-{capability}", resolved=_base_resolved(capability)
    )
    assert payload["contract_version"] == CONTRACT_VERSION


def test_adapter_artifact_json_embeds_canonical_capability_package_metadata() -> None:
    # "repair" is a real aiodoo_contract CapabilityName with family/architecture
    # resolvable from the config, so the canonical block must be present.
    payload = build_adapter_artifact_json(
        experiment_id="aiodoo-repair",
        resolved=_base_resolved("repair"),
        created_at="2026-07-19T12:00:00Z",
    )
    package_metadata = payload["capability_package_metadata"]
    assert package_metadata["capability"] == "repair"
    assert package_metadata["adapter_type"] == "repair"
    assert package_metadata["peft_type"] == "qlora"
    assert package_metadata["family"] == "qwen"
    assert package_metadata["architecture"] == "qwen"
    assert package_metadata["contract_version"] == CONTRACT_VERSION
    assert package_metadata["created_at"] == "2026-07-19T12:00:00Z"


def test_adapter_artifact_json_omits_capability_package_metadata_for_non_capability() -> None:
    # "context" is not an aiodoo_contract CapabilityName (it is not itself a
    # capability — see CONTRACT_ADOPTION.md); the canonical block is
    # intentionally omitted rather than raising or fabricating a value.
    payload = build_adapter_artifact_json(
        experiment_id="aiodoo-context", resolved=_base_resolved("context")
    )
    assert "capability_package_metadata" not in payload
    assert payload["contract_version"] == CONTRACT_VERSION


def test_merged_artifact_json_carries_contract_version() -> None:
    payload = build_merged_artifact_json(
        experiment_id="aiodoo-planner",
        resolved=_base_resolved("planner"),
        source_bundle="/tmp/bundle",
    )
    assert payload["contract_version"] == CONTRACT_VERSION
    assert payload["capability_package_metadata"]["capability"] == "planner"


def test_base_model_artifact_json_carries_contract_version() -> None:
    payload = build_base_model_artifact_json(model_id="Qwen/Qwen3-8B", model_family="qwen")
    assert payload["contract_version"] == CONTRACT_VERSION


def test_supported_odoo_versions_from_config() -> None:
    resolved = _base_resolved("repair", supported_odoo_versions=[18, 19])
    payload = build_adapter_artifact_json(experiment_id="aiodoo-repair", resolved=resolved)
    assert payload["supported_odoo_versions"] == [18, 19]


def test_merged_artifact_json_has_capability() -> None:
    payload = build_merged_artifact_json(
        experiment_id="aiodoo-planner",
        resolved=_base_resolved("planner"),
        source_bundle="/tmp/bundle",
    )
    assert payload["artifact_type"] == MERGED_PROTOCOL_ARTIFACT_TYPE
    assert payload["capability_id"] == "planner"
    assert payload["adapter_type"] == "planner"
    assert payload["source_bundle"] == "/tmp/bundle"


def test_base_model_artifact_json_has_architecture() -> None:
    payload = build_base_model_artifact_json(
        model_id="Qwen/Qwen3-8B",
        model_family="qwen",
        created_at="2026-07-19T12:00:00Z",
    )
    assert payload["artifact_type"] == "base_model"
    assert payload["architecture"] == "qwen"
    assert payload["created_at"] == "2026-07-19T12:00:00Z"
    assert payload["producer"] == "aiodoo-training"


def _write_valid_checkpoint(ckpt: Path) -> None:
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt / "adapter_model.safetensors").write_text("weights", encoding="utf-8")


def test_publish_repair_capability_package(tmp_path: Path) -> None:
    workspace = tmp_path / "AIODOO"
    workspace.mkdir()
    resolved = _base_resolved("repair")
    resolved["workspace"] = {"layout": "drive_v1", "root": str(workspace)}
    manager = ArtifactOutputManager(
        layout=ArtifactOutputLayout.for_training(workspace, "repair"),
        resolved=resolved,
    )
    ckpt = workspace / "training" / "cache" / "repair" / "checkpoints" / "checkpoint-10"
    _write_valid_checkpoint(ckpt)

    dest = manager.publish_adapter_from_checkpoint(ckpt)
    assert dest is not None
    artifact = json.loads((dest / ARTIFACT_METADATA_FILENAME).read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "coding_adapter"
    assert artifact["capability_id"] == "repair"
    assert artifact["adapter_type"] == "repair"
    assert artifact["peft_type"] == "qlora"

    summary = manager.write_experiment_summary(
        run_id="run-1",
        success=True,
        duration_seconds=1.0,
        paths={"adapter": str(dest)},
    )
    body = json.loads(summary.read_text(encoding="utf-8"))
    assert body["capability_id"] == "repair"
    assert body["paths"]["capability_package"] == str(dest)
    assert body["paths"]["artifact_json"] == str(dest / ARTIFACT_METADATA_FILENAME)
