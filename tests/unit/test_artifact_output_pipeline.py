"""Unit tests for central artifact output pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiodoo_training.application.train_orchestrator import apply_colab_path_overrides
from aiodoo_training.artifacts.cleanup import (
    cleanup_workspace,
    is_protected_path,
    scan_empty_directories,
)
from aiodoo_training.artifacts.output_layout import ArtifactOutputLayout
from aiodoo_training.artifacts.output_manager import (
    ArtifactOutputManager,
    resolve_workspace_root,
    should_use_canonical_layout,
    validate_drive_workspace_contract,
)
from aiodoo_training.artifacts.publish_contract import (
    ARTIFACT_METADATA_FILENAME,
    PublishError,
    validate_checkpoint_for_publish,
)
from aiodoo_training.exceptions import ConfigError


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "AIODOO"
    root.mkdir()
    return root


@pytest.fixture
def resolved_config(workspace: Path) -> dict:
    return {
        "name": "coding",
        "experiment": {"id": "coding", "internal_id": "EXP-0001", "stage": "coding"},
        "metadata": {"internal_id": "EXP-0001"},
        "workspace": {"layout": "drive_v1", "root": str(workspace)},
        "checkpointing": {"output_dir": "artifacts/checkpoints/coding"},
        "export": {"output_dir": "artifacts/export/coding"},
        "metrics": {"history_path": "artifacts/checkpoints/coding/metrics/history.jsonl"},
        "tracking": {"root_dir": "artifacts/tracking/coding"},
        "datasets": [{"path": "coding.jsonl", "dataset_type": "coding"}],
        "dataset_version": "v1.0.0",
        "model": {"base_model": "Qwen/Qwen3-8B", "family": "qwen"},
        "adaptation": {"adapter_type": "qlora", "strategy": "qlora"},
    }


def test_layout_paths(workspace: Path) -> None:
    layout = ArtifactOutputLayout.for_training(workspace, "coding")
    assert layout.adapter_dir == workspace / "models" / "adapters" / "aiodoo-coding"
    assert layout.merged_dir == workspace / "models" / "merged" / "aiodoo-coding"
    assert layout.export_dir == workspace / "models" / "exports" / "aiodoo-coding"
    assert layout.adapter_checkpoints_dir == (
        workspace / "training" / "cache" / "coding" / "checkpoints"
    )
    assert layout.experiment_dir == workspace / "experiments" / "coding"
    assert layout.metrics_history_path == (
        workspace / "experiments" / "coding" / "metrics" / "history.jsonl"
    )


def test_layout_normalizes_legacy_internal_id(workspace: Path) -> None:
    layout = ArtifactOutputLayout.for_training(workspace, "EXP-0001")
    assert layout.training_id == "coding"
    assert layout.adapter_id == "aiodoo-coding"


def test_manager_rewrites_config_paths(resolved_config: dict, workspace: Path) -> None:
    manager = ArtifactOutputManager.from_resolved(resolved_config)
    assert manager is not None
    updated = manager.apply_to_resolved(resolved_config)
    assert updated["checkpointing"]["output_dir"] == str(
        workspace / "training" / "cache" / "coding" / "checkpoints"
    )
    assert updated["export"]["output_dir"] == str(
        workspace / "models" / "exports" / "aiodoo-coding"
    )
    assert updated["metrics"]["history_path"] == str(
        workspace / "experiments" / "coding" / "metrics" / "history.jsonl"
    )
    assert updated["tracking"]["root_dir"] == str(
        workspace / "experiments" / "coding" / "logs" / "tracking"
    )
    assert updated["workspace"]["training_id"] == "coding"
    assert updated["workspace"]["adapter_id"] == "aiodoo-coding"


def test_should_use_canonical_layout_with_workspace_env(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    monkeypatch.setenv("AIODOO_WORKSPACE_ROOT", str(workspace))
    assert should_use_canonical_layout({}) is True
    assert resolve_workspace_root({}) == workspace


def test_drive_v1_without_workspace_raises(resolved_config: dict) -> None:
    config = dict(resolved_config)
    config["workspace"] = {"layout": "drive_v1"}
    with pytest.raises(ConfigError, match="AIODOO_WORKSPACE_ROOT"):
        validate_drive_workspace_contract(config)


def test_no_canonical_layout_without_workspace() -> None:
    config = {"name": "coding", "checkpointing": {"output_dir": "artifacts/checkpoints"}}
    assert ArtifactOutputManager.from_resolved(config) is None


def _write_valid_checkpoint(ckpt: Path) -> None:
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    (ckpt / "rng.json").write_text("{}", encoding="utf-8")
    (ckpt / "metrics.json").write_text("{}", encoding="utf-8")


def test_publish_adapter_from_checkpoint(workspace: Path, resolved_config: dict) -> None:
    manager = ArtifactOutputManager(
        layout=ArtifactOutputLayout.for_training(workspace, "coding"),
        resolved=resolved_config,
    )
    ckpt = workspace / "training" / "cache" / "coding" / "checkpoints" / "checkpoint-100"
    _write_valid_checkpoint(ckpt)

    dest = manager.publish_adapter_from_checkpoint(ckpt)
    assert dest is not None
    assert dest == workspace / "models" / "adapters" / "aiodoo-coding"
    assert (dest / "adapter_config.json").is_file()
    assert (dest / "adapter_model.safetensors").is_file()
    assert not (dest / "rng.json").exists()
    assert not (dest / "metrics.json").exists()

    artifact = json.loads((dest / ARTIFACT_METADATA_FILENAME).read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "coding_adapter"
    assert artifact["protocol_major"] == 1
    assert artifact["identifier"] == "aiodoo-coding"
    assert artifact["capability_id"] == "coding"
    assert artifact["adapter_type"] == "coding"
    assert artifact["peft_type"] == "qlora"
    assert artifact["model_family"] == "qwen"
    assert artifact["supported_odoo_versions"]
    assert artifact["created_at"]
    assert artifact["producer"] == "aiodoo-training"

    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_id"] == "coding"
    assert manifest["adapter_id"] == "aiodoo-coding"
    assert manifest["experiment_id"] == "coding"
    assert manifest["capability_id"] == "coding"


def test_publish_rejects_invalid_checkpoint(workspace: Path, resolved_config: dict) -> None:
    manager = ArtifactOutputManager(
        layout=ArtifactOutputLayout.for_training(workspace, "coding"),
        resolved=resolved_config,
    )
    ckpt = workspace / "training" / "cache" / "coding" / "checkpoints" / "checkpoint-1"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PublishError, match="adapter weights"):
        manager.publish_adapter_from_checkpoint(ckpt)


def test_validate_checkpoint_for_publish() -> None:
    with pytest.raises(PublishError):
        validate_checkpoint_for_publish(Path("/nonexistent"))


def test_atomic_publish_replaces_existing(workspace: Path, resolved_config: dict) -> None:
    manager = ArtifactOutputManager(
        layout=ArtifactOutputLayout.for_training(workspace, "coding"),
        resolved=resolved_config,
    )
    ckpt1 = workspace / "training" / "cache" / "coding" / "checkpoints" / "checkpoint-50"
    _write_valid_checkpoint(ckpt1)
    manager.publish_adapter_from_checkpoint(ckpt1)

    ckpt2 = workspace / "training" / "cache" / "coding" / "checkpoints" / "checkpoint-100"
    _write_valid_checkpoint(ckpt2)
    (ckpt2 / "adapter_model.safetensors").write_text("new-weights", encoding="utf-8")
    manager.publish_adapter_from_checkpoint(ckpt2)

    assert (manager.layout.adapter_dir / "adapter_model.safetensors").read_text() == "new-weights"


def test_publish_base_model_artifact(workspace: Path, resolved_config: dict) -> None:
    model_dir = workspace / "model-cache" / "Qwen__Qwen3-8B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    config = dict(resolved_config)
    config["model"] = {"base_model": "Qwen/Qwen3-8B", "local_path": str(model_dir)}
    manager = ArtifactOutputManager.from_resolved(config)
    assert manager is not None
    path = manager.publish_base_model_artifact(model_dir)
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "base_model"
    assert payload["protocol_major"] == 1
    assert payload["architecture"]
    assert payload["created_at"]
    assert payload["producer"] == "aiodoo-training"


def test_cleanup_protected_paths(workspace: Path) -> None:
    adapter = workspace / "models" / "adapters" / "aiodoo-coding"
    adapter.mkdir(parents=True)
    assert is_protected_path(adapter, workspace)

    empty_cache = workspace / "training" / "cache" / "coding" / "checkpoints" / "empty"
    empty_cache.mkdir(parents=True)
    found = scan_empty_directories(workspace, workspace_root=workspace)
    assert empty_cache in found
    assert adapter not in found


def test_cleanup_does_not_delete_protected_empty_adapter_dir(workspace: Path) -> None:
    empty_adapter = workspace / "models" / "adapters" / "aiodoo-coding"
    empty_adapter.mkdir(parents=True)
    report = cleanup_workspace(workspace, dry_run=False, delete=True)
    assert empty_adapter not in report.deleted
    assert empty_adapter.exists()


def test_cleanup_delete_removes_empty_cache_dirs(workspace: Path) -> None:
    empty = workspace / "training" / "cache" / "coding" / "checkpoints" / "orphan"
    empty.mkdir(parents=True)
    report = cleanup_workspace(workspace, dry_run=False, delete=True)
    assert empty in report.deleted
    assert not empty.exists()


def test_apply_colab_overrides_with_workspace(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    monkeypatch.setenv("AIODOO_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("AIODOO_COLAB_MODEL_PATH", "/content/model")
    monkeypatch.setenv("AIODOO_COLAB_DATASET_PATH", str(workspace / "datasets" / "v1.0.0"))
    resolved = {
        "name": "coding",
        "experiment": {"id": "coding"},
        "workspace": {"layout": "drive_v1"},
        "checkpointing": {"output_dir": "artifacts/checkpoints/coding"},
        "export": {"output_dir": "artifacts/export/coding"},
    }
    out = apply_colab_path_overrides(resolved)
    assert "artifacts/checkpoints" not in out["checkpointing"]["output_dir"]
    assert str(workspace) in out["checkpointing"]["output_dir"]
    assert out["model"]["local_path"] == "/content/model"


def test_colab_legacy_output_env_ignored(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    monkeypatch.setenv("AIODOO_WORKSPACE_ROOT", str(workspace))
    legacy_ckpt = workspace / "legacy" / "ckpt"
    monkeypatch.setenv("AIODOO_COLAB_CHECKPOINTS_OUTPUT", str(legacy_ckpt))
    resolved = {
        "name": "coding",
        "experiment": {"id": "coding"},
        "workspace": {"layout": "drive_v1"},
        "checkpointing": {"output_dir": "artifacts/checkpoints/coding"},
    }
    out = apply_colab_path_overrides(resolved)
    assert out["checkpointing"]["output_dir"] == str(
        workspace / "training" / "cache" / "coding" / "checkpoints"
    )
    assert legacy_ckpt.as_posix() not in out["checkpointing"]["output_dir"]
