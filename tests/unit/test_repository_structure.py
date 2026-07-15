"""Tests that required repository structure exists."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


REQUIRED_DIRS = [
    "docs/adr",
    "configs/experiments",
    "configs/models",
    "configs/datasets",
    "configs/adaptation",
    "configs/training",
    "configs/evaluation",
    "configs/export",
    "configs/schemas",
    "requirements",
    "tests/unit",
    "tests/integration",
    "tests/contract",
    "tests/golden",
    "tests/fixtures",
    ".github/workflows",
    "aiodoo_training/domain",
    "aiodoo_training/ports",
    "aiodoo_training/registries",
    "aiodoo_training/builders",
    "aiodoo_training/factories",
    "aiodoo_training/config",
    "aiodoo_training/pipeline",
    "aiodoo_training/determinism",
    "aiodoo_training/cli",
    "aiodoo_training/application",
    "aiodoo_training/datasets",
    "aiodoo_training/tokenization",
    "aiodoo_training/models",
    "aiodoo_training/adaptation",
    "aiodoo_training/packing",
    "aiodoo_training/curriculum",
    "aiodoo_training/training",
    "aiodoo_training/checkpointing",
    "aiodoo_training/evaluation",
    "aiodoo_training/export",
    "aiodoo_training/tracking",
    "aiodoo_training/infrastructure/huggingface",
    "aiodoo_training/infrastructure/peft",
    "aiodoo_training/infrastructure/accelerate",
    "aiodoo_training/infrastructure/resources",
    "aiodoo_training/infrastructure/storage",
    "aiodoo_training/infrastructure/trackers",
]

REQUIRED_FILES = [
    "pyproject.toml",
    "requirements.txt",
    "requirements/base.txt",
    "requirements/train.txt",
    "requirements/dev.txt",
    "README.md",
    "LICENSE",
    "docs/architecture.md",
    "docs/adr/0001-immutable-domain.md",
    "docs/adr/0002-ports-over-frameworks.md",
    "docs/adr/0003-pipeline-architecture.md",
    "docs/adr/0004-experiment-fingerprint.md",
    "docs/adr/0005-adaptation-separation.md",
    "docs/adr/0006-repository-boundaries.md",
    "docs/adr/0008-dataset-session-and-chat-templates.md",
    "docs/adr/0009-resource-management.md",
    "docs/adr/0010-phase2-model-adaptation.md",
    "docs/adr/0011-phase2-hardening.md",
    "docs/adr/0012-phase2-freeze.md",
    "docs/architecture_invariants.md",
    "docs/frozen_public_contracts.md",
    "docs/phase3-training-engine-architecture.md",
    "docs/adr/0013-phase3-training-engine.md",
    ".github/workflows/ci.yml",
    "configs/experiments/example.yaml",
    "validate_config.py",
    "fingerprint.py",
    "doctor.py",
    "train.py",
    "resume.py",
    "evaluate.py",
    "export.py",
    "merge.py",
    "build_training.py",
    "build_config.py",
    "prepare_dataset.py",
    "tests/run_tests.py",
]

FORBIDDEN_FILES = [
    "setup.py",
    "setup.cfg",
    "aiodoo_training/cli/main.py",
]


@pytest.mark.parametrize("relative", REQUIRED_DIRS)
def test_required_directory_exists(relative: str) -> None:
    assert (ROOT / relative).is_dir(), f"Missing directory: {relative}"


@pytest.mark.parametrize("relative", REQUIRED_FILES)
def test_required_file_exists(relative: str) -> None:
    assert (ROOT / relative).is_file(), f"Missing file: {relative}"


@pytest.mark.parametrize("relative", FORBIDDEN_FILES)
def test_packaging_and_dead_files_absent(relative: str) -> None:
    assert not (ROOT / relative).exists(), f"Must not exist: {relative}"


def test_no_egg_info_directory() -> None:
    assert not (ROOT / "aiodoo_training.egg-info").exists()


def test_pyproject_is_tooling_only() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[build-system]" not in text
    assert "setuptools" not in text
    assert "[project]" not in text
    assert "console_scripts" not in text
    assert "[tool.ruff]" in text
    assert "[tool.mypy]" in text
