"""Tokenization pipeline orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from aiodoo_training.datasets.caching import (
    DatasetCache,
    TokenCacheKey,
    fingerprint_tokenization_config,
)
from aiodoo_training.domain.examples import TokenBatch, TokenizationConfig, TrainingExample
from aiodoo_training.ports.chat_template import ChatTemplate
from aiodoo_training.ports.tokenizer import TokenizerPort
from aiodoo_training.registries import chat_template_registry
from aiodoo_training.tokenization.fingerprints import fingerprint_masking_policy


class TokenizationPipeline:
    """Deterministic examples → TokenBatch pipeline with optional disk cache."""

    def __init__(
        self,
        tokenizer: TokenizerPort,
        *,
        template: ChatTemplate | None = None,
        config: TokenizationConfig | None = None,
        cache: DatasetCache | None = None,
        dataset_fingerprint: str = "",
    ) -> None:
        self._tokenizer = tokenizer
        self._config = config or TokenizationConfig()
        self._template = template
        self._cache = cache
        self._dataset_fingerprint = dataset_fingerprint
        if self._template is not None:
            # Allow backends that expose injectable template/config attributes.
            if hasattr(self._tokenizer, "_template"):
                self._tokenizer._template = self._template
            if hasattr(self._tokenizer, "_config"):
                self._tokenizer._config = self._config

    def run(self, examples: Sequence[TrainingExample]) -> TokenBatch:
        key = self._cache_key()
        if self._cache is not None and key is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        batch = self._tokenizer.encode_examples(examples)
        if self._cache is not None and key is not None:
            self._cache.put(key, batch)
        return batch

    def _cache_key(self) -> TokenCacheKey | None:
        if not self._dataset_fingerprint:
            return None
        template = self._template
        if template is None and chat_template_registry.exists(self._config.chat_template_name):
            template = chat_template_registry.get(self._config.chat_template_name)()
        if template is None:
            return None
        return TokenCacheKey(
            dataset_fingerprint=self._dataset_fingerprint,
            tokenizer_fingerprint=self._tokenizer.fingerprint(),
            chat_template_fingerprint=template.fingerprint(),
            masking_policy_fingerprint=fingerprint_masking_policy(
                mask_prompt=self._config.mask_prompt,
                ignore_index=self._config.ignore_index,
            ),
            config_fingerprint=fingerprint_tokenization_config(self._config),
        )


def default_cache(root: Path | None = None) -> DatasetCache:
    return DatasetCache(root or Path(".aiodoo_training_cache") / "tokens")
