"""Boundary tests: Torch / Transformers / PEFT stay inside infrastructure."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "aiodoo_training"

FORBIDDEN_IMPORTS = frozenset(
    {
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "mlflow",
        "wandb",
        "tensorboard",
        "deepspeed",
        "nccl",
        "torch_xla",
    }
)

ALLOWED_PREFIX = ROOT / "infrastructure"


def _iter_py_files() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.py") if p.is_file())


def _is_under_infrastructure(path: Path) -> bool:
    try:
        path.relative_to(ALLOWED_PREFIX)
        return True
    except ValueError:
        return False


def _literal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.split(".")[0]
    return None


def find_forbidden_framework_refs(source: str, *, filename: str = "<memory>") -> set[str]:
    """
    Detect forbidden framework imports / dynamic loads in ``source``.

    Catches:
    - ``import torch`` / ``from transformers import ...``
    - ``__import__("torch")``
    - ``importlib.import_module("peft")``
    """
    tree = ast.parse(source, filename=filename)
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_IMPORTS:
                found.add(root)
        elif isinstance(node, ast.Call):
            func = node.func
            # __import__("torch")
            if isinstance(func, ast.Name) and func.id == "__import__" and node.args:
                name = _literal_name(node.args[0])
                if name in FORBIDDEN_IMPORTS:
                    found.add(name)
            # importlib.import_module("torch")
            if isinstance(func, ast.Attribute) and func.attr == "import_module" and node.args:
                name = _literal_name(node.args[0])
                if name in FORBIDDEN_IMPORTS:
                    found.add(name)
            # import_module("torch") after from importlib import import_module
            if isinstance(func, ast.Name) and func.id == "import_module" and node.args:
                name = _literal_name(node.args[0])
                if name in FORBIDDEN_IMPORTS:
                    found.add(name)

    return found


def test_framework_imports_confined_to_infrastructure() -> None:
    violations: list[str] = []
    for path in _iter_py_files():
        if _is_under_infrastructure(path):
            continue
        leaked = find_forbidden_framework_refs(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
        if leaked:
            violations.append(f"{path.relative_to(ROOT.parent)}: {sorted(leaked)}")
    assert not violations, "Framework imports leaked outside infrastructure:\n" + "\n".join(
        violations
    )


def test_boundary_scanner_detects_static_and_dynamic_imports() -> None:
    samples = {
        "import torch\n": {"torch"},
        "from transformers import AutoModel\n": {"transformers"},
        "import peft as p\n": {"peft"},
        'x = __import__("bitsandbytes")\n': {"bitsandbytes"},
        'import importlib\nimportlib.import_module("accelerate")\n': {"accelerate"},
        'from importlib import import_module\nimport_module("torch")\n': {"torch"},
        "from aiodoo_training.domain import Precision\n": set(),
    }
    for source, expected in samples.items():
        assert find_forbidden_framework_refs(source) == expected


def test_infrastructure_may_reference_framework_names() -> None:
    """Sanity: scanner still reports names; infra paths are exempted by path check."""
    assert _is_under_infrastructure(ALLOWED_PREFIX / "peft" / "__init__.py")
    assert not _is_under_infrastructure(ROOT / "domain" / "model_info.py")


def test_boundary_scanner_would_fail_on_leaked_module(tmp_path: Path) -> None:
    """Demonstrate that a leaked import outside infra is detected."""
    leaked = textwrap.dedent(
        """
        def load():
            import torch
            return torch
        """
    )
    path = tmp_path / "leak.py"
    path.write_text(leaked, encoding="utf-8")
    found = find_forbidden_framework_refs(path.read_text(encoding="utf-8"), filename=str(path))
    assert found == {"torch"}
    # Guardrail: if such a file lived under aiodoo_training (non-infra), CI would fail.
    with pytest.raises(AssertionError):
        if found:
            raise AssertionError(f"leak: {sorted(found)}")
