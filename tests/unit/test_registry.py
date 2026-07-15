"""Unit tests for the generic Registry."""

import pytest

from aiodoo_training.exceptions import RegistryError
from aiodoo_training.registries import Registry


def test_register_get_exists_list_and_repr() -> None:
    registry: Registry[str] = Registry("demo")
    registry.register("a", "alpha")
    registry.register("b", "beta")

    assert registry.exists("a")
    assert registry.get("a") == "alpha"
    assert registry.list() == ("a", "b")
    assert len(registry) == 2
    assert "a" in registry
    assert "frozen" not in repr(registry)
    assert "mutable" in repr(registry)


def test_duplicate_register_raises_with_known_keys() -> None:
    registry: Registry[int] = Registry("nums")
    registry.register("x", 1)
    with pytest.raises(RegistryError, match="Duplicate"):
        registry.register("x", 2)


def test_overwrite_allowed_when_requested() -> None:
    registry: Registry[int] = Registry("nums")
    registry.register("x", 1)
    registry.register("x", 2, overwrite=True)
    assert registry.get("x") == 2


def test_freeze_is_idempotent_and_blocks_mutation() -> None:
    registry: Registry[str] = Registry("frozen")
    registry.register("k", "v")
    registry.freeze()
    registry.freeze()
    assert registry.is_frozen
    assert "frozen" in repr(registry)

    with pytest.raises(RegistryError, match="frozen"):
        registry.register("other", "x")
    with pytest.raises(RegistryError, match="frozen"):
        registry.clear()


def test_clear_and_unknown_key_message() -> None:
    registry: Registry[str] = Registry("tmp")
    registry.register("k", "v")
    registry.clear()
    assert len(registry) == 0
    with pytest.raises(RegistryError, match=r"Known keys: \(none\)"):
        registry.get("k")


def test_empty_key_and_empty_name_rejected() -> None:
    with pytest.raises(RegistryError, match="name"):
        Registry("  ")
    registry: Registry[str] = Registry("tmp")
    with pytest.raises(RegistryError, match="non-empty"):
        registry.register("  ", "v")
