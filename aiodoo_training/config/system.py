"""Configuration loading, composition, validation, resolution, and hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aiodoo_training.domain.identifiers import ExperimentId
from aiodoo_training.exceptions import ConfigError

INCLUDE_KEY = "include"


class RawExperimentModel(BaseModel):
    """
    Lenient Pydantic model for Phase 0 config validation.

    Field shapes harden in later phases as domain mapping is completed.
    Unknown keys are allowed so fragment composition remains flexible.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    seed: int = 42
    model: dict[str, Any] = Field(default_factory=dict)
    datasets: dict[str, Any] | list[Any] = Field(default_factory=dict)
    adaptation: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    packing: dict[str, Any] = Field(default_factory=dict)
    curriculum: dict[str, Any] = Field(default_factory=dict)
    checkpointing: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    export: dict[str, Any] = Field(default_factory=dict)
    tracking: dict[str, Any] = Field(default_factory=dict)
    determinism: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    distributed: dict[str, Any] = Field(default_factory=dict)


class ConfigLoader:
    """Load a single YAML configuration document from disk."""

    def load(self, path: Path) -> dict[str, Any]:
        """
        Load ``path`` into a dictionary.

        Raises:
            ConfigError: if the file is missing, unreadable, or not a mapping.
        """
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        if not path.is_file():
            raise ConfigError(f"Config path is not a file: {path}")
        try:
            raw = path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"Failed to read config {path}: {exc}") from exc
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError(f"Config root must be a mapping: {path}")
        return cast(dict[str, Any], data)


class ConfigComposer:
    """
    Compose configuration from multiple YAML fragments.

    Supports an ``include`` list of relative paths merged left-to-right.
    Nested mappings are deep-merged; scalars and lists are replaced.
    The including document wins on conflicts after includes are applied.
    """

    def __init__(self, loader: ConfigLoader | None = None) -> None:
        self._loader = loader or ConfigLoader()

    def compose(self, path: Path) -> dict[str, Any]:
        """Compose ``path`` and any included fragments into one mapping."""
        return self._compose(path, stack=())

    def _compose(self, path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        resolved = path.resolve()
        if resolved in stack:
            cycle = " -> ".join(str(p) for p in (*stack, resolved))
            raise ConfigError(f"Circular config include detected: {cycle}")

        document = dict(self._loader.load(resolved))
        includes = document.pop(INCLUDE_KEY, [])
        if includes is None:
            includes = []
        if not isinstance(includes, list):
            raise ConfigError(f"'{INCLUDE_KEY}' must be a list in {resolved}")

        merged: dict[str, Any] = {}
        base_dir = resolved.parent
        for entry in includes:
            if not isinstance(entry, str) or not entry.strip():
                raise ConfigError(
                    f"Include entries must be non-empty strings in {resolved}; got {entry!r}"
                )
            include_path = (base_dir / entry).resolve()
            fragment = self._compose(include_path, (*stack, resolved))
            merged = deep_merge(merged, fragment)

        return deep_merge(merged, document)


class ConfigValidator:
    """Validate composed configuration against the Phase 0 schema."""

    def validate(self, data: dict[str, Any]) -> RawExperimentModel:
        """
        Validate ``data``.

        Raises:
            ConfigError: wrapping pydantic validation failures with a clear message.
        """
        try:
            return RawExperimentModel.model_validate(data)
        except PydanticValidationError as exc:
            raise ConfigError(f"Config validation failed:\n{exc}") from exc


class ConfigResolver:
    """
    Resolve path-like values relative to a configuration base directory.

    Phase 0 resolves well-known path keys without loading external artifacts.
    Resolved absolute paths are for runtime I/O only — they must not feed
    experiment identity hashing (see :class:`ConfigSystem`).
    """

    PATH_KEYS = frozenset(
        {
            "path",
            "local_path",
            "output_dir",
            "resume_from",
            "tracking_uri",
        }
    )

    def resolve(self, data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
        """Return a deep copy of ``data`` with relative paths made absolute."""
        return cast(dict[str, Any], self._resolve_node(data, base_dir.resolve()))

    def _resolve_node(self, node: Any, base_dir: Path) -> Any:
        if isinstance(node, dict):
            result: dict[str, Any] = {}
            for key, value in node.items():
                if key in self.PATH_KEYS and isinstance(value, str) and value.strip():
                    path = Path(value)
                    result[key] = str(path if path.is_absolute() else (base_dir / path).resolve())
                else:
                    result[key] = self._resolve_node(value, base_dir)
            return result
        if isinstance(node, list):
            return [self._resolve_node(item, base_dir) for item in node]
        return node


class ConfigHasher:
    """Canonical serialization and fingerprint generation for experiments."""

    def canonical_json(self, data: dict[str, Any]) -> str:
        """
        Serialize ``data`` to a canonical JSON string.

        Keys are sorted; separators are compact; UTF-8 is preserved.
        """
        try:
            return json.dumps(
                data,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Config is not canonically serializable (required for hashing): {exc}"
            ) from exc

    def hash(self, data: dict[str, Any]) -> str:
        """Return a SHA-256 hex digest of the canonical form of ``data``."""
        payload = self.canonical_json(data).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def experiment_id(self, data: dict[str, Any], *, prefix: str = "exp") -> ExperimentId:
        """Derive a deterministic ExperimentId from composed (portable) config data."""
        digest = self.hash(data)
        return ExperimentId(value=f"{prefix}_{digest[:16]}")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two dictionaries without mutating inputs."""
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = value
    return result


class ConfigSystem:
    """
    Facade wiring loader → composer → validator → resolver → hasher.

    Experiment identity is derived from the **composed** (portable) mapping.
    Path resolution produces a separate runtime mapping and must not affect
    :class:`ExperimentId` stability across machines.
    """

    def __init__(
        self,
        loader: ConfigLoader | None = None,
        composer: ConfigComposer | None = None,
        validator: ConfigValidator | None = None,
        resolver: ConfigResolver | None = None,
        hasher: ConfigHasher | None = None,
    ) -> None:
        self.loader = loader or ConfigLoader()
        self.composer = composer or ConfigComposer(self.loader)
        self.validator = validator or ConfigValidator()
        self.resolver = resolver or ConfigResolver()
        self.hasher = hasher or ConfigHasher()

    def load_experiment(
        self,
        path: Path,
    ) -> tuple[RawExperimentModel, ExperimentId, dict[str, Any]]:
        """
        Compose, validate, resolve, and fingerprint an experiment config file.

        Returns:
            validated model, portable experiment id, and path-resolved mapping.
        """
        composed = self.composer.compose(path)
        model = self.validator.validate(composed)
        resolved = self.resolver.resolve(composed, path.resolve().parent)
        experiment_id = self.hasher.experiment_id(composed)
        return model, experiment_id, resolved
