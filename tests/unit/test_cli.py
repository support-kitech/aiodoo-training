"""Root entry-script foundation tests."""

from pathlib import Path

import pytest

import build_config
import build_training
import doctor
import fingerprint
import train
import validate_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "configs" / "experiments" / "example.yaml"


def test_validate_config_script(capsys: pytest.CaptureFixture[str]) -> None:
    assert validate_config.main(["--config", str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "Config OK" in out
    assert "config_experiment_id:" in out


def test_fingerprint_script(capsys: pytest.CaptureFixture[str]) -> None:
    assert fingerprint.main(["--config", str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "digest:" in out
    assert "package_digest:" in out


def test_doctor_script(capsys: pytest.CaptureFixture[str]) -> None:
    assert doctor.main() == 0
    out = capsys.readouterr().out
    assert "aiodoo-training:" in out
    assert "repository-root scripts" in out


def test_build_config_script(capsys: pytest.CaptureFixture[str]) -> None:
    assert build_config.main(["--config", str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "resolved" in out
    assert "config_experiment_id:" in out


def test_build_training_validate_only() -> None:
    assert build_training.main(["--config", str(EXAMPLE), "--validate-only"]) == 0


def test_train_not_implemented() -> None:
    assert train.main(["--config", str(EXAMPLE)]) == 2
