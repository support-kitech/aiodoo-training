"""Generic freezable registry infrastructure."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType

from aiodoo_training.exceptions import RegistryError


class Registry[T]:
    """
    Typed, freezable name → value registry.

    Lifecycle:
        Create → register / clear → freeze → lookup only

    After :meth:`freeze`, mutation attempts raise :class:`RegistryError`.
    Calling :meth:`freeze` again is a no-op (idempotent).
    """

    def __init__(self, name: str = "registry") -> None:
        if not name or not name.strip():
            raise RegistryError("Registry name must be a non-empty string.")
        self._name = name.strip()
        self._items: dict[str, T] = {}
        self._frozen = False

    @property
    def name(self) -> str:
        """Stable registry identity used in error messages."""
        return self._name

    @property
    def is_frozen(self) -> bool:
        """True when further registrations are rejected."""
        return self._frozen

    def _assert_mutable(self) -> None:
        if self._frozen:
            raise RegistryError(
                f"Cannot mutate frozen registry '{self._name}'. "
                "Create a new Registry instance if reconfiguration is required."
            )

    def register(self, key: str, value: T, *, overwrite: bool = False) -> None:
        """
        Register ``value`` under ``key``.

        Raises:
            RegistryError: if the registry is frozen, ``key`` is empty, or
                ``key`` already exists and ``overwrite`` is False.
        """
        self._assert_mutable()
        if not key or not key.strip():
            raise RegistryError(f"Registry key must be a non-empty string in '{self._name}'.")
        normalized = key.strip()
        if normalized in self._items and not overwrite:
            raise RegistryError(
                f"Duplicate registration for key '{normalized}' in '{self._name}'. "
                f"Known keys: {self._known_keys()}."
            )
        self._items[normalized] = value

    def get(self, key: str) -> T:
        """
        Return the value for ``key``.

        Raises:
            RegistryError: if ``key`` is not registered.
        """
        try:
            return self._items[key]
        except KeyError as exc:
            raise RegistryError(
                f"Unknown key '{key}' in registry '{self._name}'. Known keys: {self._known_keys()}."
            ) from exc

    def exists(self, key: str) -> bool:
        """Return True if ``key`` is registered."""
        return key in self._items

    def list(self) -> tuple[str, ...]:
        """Return registered keys in sorted order (deterministic)."""
        return tuple(sorted(self._items.keys()))

    def items(self) -> Mapping[str, T]:
        """Return an immutable view of registry contents."""
        return MappingProxyType(self._items)

    def clear(self) -> None:
        """Remove all registrations. Raises if frozen."""
        self._assert_mutable()
        self._items.clear()

    def freeze(self) -> None:
        """Freeze the registry against further mutation (idempotent)."""
        self._frozen = True

    def _known_keys(self) -> str:
        keys = self.list()
        return ", ".join(keys) if keys else "(none)"

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.list())

    def __repr__(self) -> str:
        state = "frozen" if self._frozen else "mutable"
        return f"Registry(name={self._name!r}, size={len(self)}, state={state})"
