"""Deterministic on-disk cache for tokenized batches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from aiodoo_training.domain.examples import TokenBatch, TokenizationConfig
from aiodoo_training.exceptions import DomainError


@dataclass(frozen=True, slots=True)
class TokenCacheKey:
    """Composite cache key — any field change invalidates the entry."""

    dataset_fingerprint: str
    tokenizer_fingerprint: str
    chat_template_fingerprint: str
    masking_policy_fingerprint: str
    config_fingerprint: str

    def digest(self) -> str:
        material = "|".join(
            [
                self.dataset_fingerprint,
                self.tokenizer_fingerprint,
                self.chat_template_fingerprint,
                self.masking_policy_fingerprint,
                self.config_fingerprint,
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


def fingerprint_tokenization_config(config: TokenizationConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DatasetCache:
    """
    Filesystem cache for TokenBatch objects.

    Stores JSON sidecars keyed by TokenCacheKey.digest().
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def get(self, key: TokenCacheKey) -> TokenBatch | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return TokenBatch(
                example_ids=tuple(raw["example_ids"]),
                input_ids=tuple(tuple(row) for row in raw["input_ids"]),
                attention_mask=tuple(tuple(row) for row in raw["attention_mask"]),
                labels=tuple(tuple(row) for row in raw["labels"]),
                metadata=MappingProxyType(raw.get("metadata", {})),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DomainError(f"Corrupt token cache entry {path}: {exc}") from exc

    def put(self, key: TokenCacheKey, batch: TokenBatch) -> Path:
        path = self._path_for(key)
        payload: dict[str, Any] = {
            "key": key.digest(),
            "example_ids": list(batch.example_ids),
            "input_ids": [list(row) for row in batch.input_ids],
            "attention_mask": [list(row) for row in batch.attention_mask],
            "labels": [list(row) for row in batch.labels],
            "metadata": dict(batch.metadata),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path

    def invalidate(self, key: TokenCacheKey) -> None:
        path = self._path_for(key)
        if path.exists():
            path.unlink()

    def _path_for(self, key: TokenCacheKey) -> Path:
        return self._root / f"{key.digest()}.json"
