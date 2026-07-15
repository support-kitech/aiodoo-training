"""Public train execution path — config → pipeline → ExecutionResult."""

from __future__ import annotations

from pathlib import Path

import pytest

import train
from aiodoo_training.application import (
    ExecutionResult,
    run_train_from_config,
    train_exit_code,
)
from aiodoo_training.bootstrap import bootstrap_phase7
from aiodoo_training.config import ConfigSystem, to_experiment_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "configs" / "experiments" / "example.yaml"


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase7(overwrite=True)


def _write_stub_experiment(tmp_path: Path) -> Path:
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        f"""
schema_version: "1.0"
name: stub-cli-train
seed: 7

model:
  identifier: fixture/stub-lm
  family: qwen
  precision: fp32
  backend: stub

adaptation:
  adapter_type: lora
  rank: 8
  alpha: 16
  target_modules:
    - q_proj

training:
  backend: stub
  learning_rate: 0.0002
  max_steps: 3
  per_device_batch_size: 1

execution:
  device:
    preferred: cpu
    allow_cpu_fallback: true
  precision:
    compute: fp32
  accelerator: none

checkpointing:
  output_dir: {tmp_path / "checkpoints"}
  save_steps: 100
  save_total_limit: 2

evaluation:
  enabled: false

export:
  enabled: false
  output_dir: {tmp_path / "export"}

tracking:
  tracker_type: null

determinism:
  seed: 7
""".strip(),
        encoding="utf-8",
    )
    return cfg


def test_to_experiment_config_from_example() -> None:
    system = ConfigSystem()
    _model, experiment_id, resolved = system.load_experiment(EXAMPLE)
    composed = system.composer.compose(EXAMPLE)
    cfg = to_experiment_config(resolved, experiment_id=experiment_id, composed=composed)
    assert cfg.name == "example-phase0"
    assert cfg.model.identifier == "Qwen/Qwen2.5-Coder-0.5B"
    assert cfg.adaptation.rank == 8
    assert cfg.optimization.learning_rate == 0.0002
    assert cfg.experiment_id == experiment_id


def test_run_train_from_config_stub_succeeds(tmp_path: Path) -> None:
    config_path = _write_stub_experiment(tmp_path)
    result = run_train_from_config(config_path)
    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.error is None
    assert result.duration_seconds >= 0.0
    assert result.checkpoint_path is not None
    assert result.adapter_path is not None
    assert train_exit_code(result) == 0


def test_train_script_stub_succeeds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = _write_stub_experiment(tmp_path)
    assert train.main(["--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "success: True" in out
    assert "duration_seconds:" in out


def test_train_script_example_stub_backend(capsys: pytest.CaptureFixture[str]) -> None:
    """example.yaml defaults model.backend to stub — should complete without NIE."""
    assert train.main(["--config", str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "success: True" in out


def test_apply_colab_path_overrides_joins_dataset_root_and_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aiodoo_training.application.train_orchestrator import apply_colab_path_overrides

    dataset_root = tmp_path / "datasets" / "v1.0.0"
    dataset_root.mkdir(parents=True)
    monkeypatch.setenv("AIODOO_COLAB_DATASET_PATH", str(dataset_root))

    resolved = {
        "datasets": {
            "datasets": [
                {"path": "coding_v1_0.jsonl", "dataset_type": "coding"},
                {"path": "/workspace/legacy/other.jsonl"},
            ]
        }
    }
    out = apply_colab_path_overrides(resolved)
    entries = out["datasets"]["datasets"]
    assert entries[0]["path"] == str(dataset_root / "coding_v1_0.jsonl")
    assert entries[1]["path"] == str(dataset_root / "other.jsonl")
