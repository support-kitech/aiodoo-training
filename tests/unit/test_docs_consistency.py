"""Freeze / docs consistency checks for v1.0.0 release readiness."""

from __future__ import annotations

from pathlib import Path

from aiodoo_training import __version__

ROOT = Path(__file__).resolve().parents[2]


def test_package_version_is_v1() -> None:
    assert __version__ == "1.0.0"


def test_changelog_mentions_v1() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [1.0.0]" in text


def test_coverage_floor_is_eighty() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "fail_under = 80" in text
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--fail-under=80" in ci


def test_freeze_docs_exist() -> None:
    for rel in (
        "docs/repository_freeze.md",
        "docs/MAINTENANCE.md",
        "docs/release_checklist.md",
        "docs/adr/0023-repository-freeze-v1.md",
        "CONTRIBUTING.md",
    ):
        assert (ROOT / rel).is_file(), rel
