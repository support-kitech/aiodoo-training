"""Phase 4 boundary tests — framework isolation and port placement."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.test_framework_boundaries import (
    FORBIDDEN_IMPORTS,
    find_forbidden_framework_refs,
)

ROOT = Path(__file__).resolve().parents[2] / "aiodoo_training"
INFRA = ROOT / "infrastructure"
PORTS = ROOT / "ports"

PHASE4_APPLICATION_DIRS = (
    ROOT / "domain",
    ROOT / "evaluation",
    ROOT / "export",
)


def _iter_py_files_under(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        files.extend(sorted(p for p in root.rglob("*.py") if p.is_file()))
    return files


def test_phase4_application_layers_avoid_framework_imports() -> None:
    violations: list[str] = []
    for path in _iter_py_files_under(*PHASE4_APPLICATION_DIRS):
        leaked = find_forbidden_framework_refs(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
        if leaked:
            violations.append(f"{path.relative_to(ROOT.parent)}: {sorted(leaked)}")
    msg = "Framework imports leaked into Phase 4 application layers:\n"
    assert not violations, msg + "\n".join(violations)


def test_evaluator_exporter_ports_live_in_ports_package() -> None:
    trainer_port = (PORTS / "trainer.py").read_text(encoding="utf-8")
    assert "class Evaluator" in trainer_port
    assert "class Exporter" in trainer_port


@pytest.mark.parametrize(
    ("relative", "symbol"),
    [
        ("stub/evaluator.py", "class StubEvaluator"),
        ("stub/exporter.py", "class StubExporter"),
        ("huggingface/evaluator.py", "class HFEvaluator"),
        ("huggingface/exporter.py", "class HFExporter"),
    ],
)
def test_concrete_evaluator_exporter_under_infrastructure(relative: str, symbol: str) -> None:
    path = INFRA / relative
    assert path.is_file(), f"Missing infrastructure adapter: {relative}"
    source = path.read_text(encoding="utf-8")
    assert symbol in source


def test_forbidden_import_set_covers_phase4_targets() -> None:
    assert FORBIDDEN_IMPORTS >= {"torch", "transformers", "peft", "bitsandbytes", "accelerate"}
