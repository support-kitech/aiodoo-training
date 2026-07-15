"""Tokenizer implementations: HuggingFace adapter + deterministic stub."""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

from aiodoo_training.domain.examples import TokenBatch, TokenizationConfig, TrainingExample
from aiodoo_training.domain.refs import ModelRef
from aiodoo_training.exceptions import DomainError
from aiodoo_training.ports.chat_template import ChatTemplate
from aiodoo_training.ports.tokenizer import TokenizerPort
from aiodoo_training.registries import chat_template_registry, tokenizer_registry
from aiodoo_training.tokenization.fingerprints import fingerprint_tokenizer_identity
from aiodoo_training.tokenization.masking import apply_assistant_only_mask, pad_sequences


class DeterministicStubTokenizer(TokenizerPort):
    """
    Offline tokenizer for tests and CPU CI — no model weights required.

    Encodes UTF-8 bytes as integer ids (shifted), which is enough to verify
    masking, padding, caching, and pipeline determinism.
    """

    def __init__(
        self,
        *,
        template: ChatTemplate | None = None,
        config: TokenizationConfig | None = None,
        pad_id: int = 0,
    ) -> None:
        self._template = template
        self._config = config or TokenizationConfig()
        self._pad_id = pad_id
        self._loaded = False
        self._name = "stub"

    def load(self, model_ref: ModelRef) -> None:
        self._name = f"stub:{model_ref.identifier}"
        if self._template is None:
            self._template = self._resolve_template(self._config.chat_template_name)
        self._loaded = True

    def vocabulary_size(self) -> int:
        return 259  # 0 pad + 1..258 byte values

    def fingerprint(self) -> str:
        return fingerprint_tokenizer_identity(self._name, vocab_size=self.vocabulary_size())

    def encode_examples(self, examples: Sequence[TrainingExample]) -> TokenBatch:
        self._ensure_loaded()
        assert self._template is not None
        config = self._config
        full_seqs: list[list[int]] = []
        label_seqs: list[list[int]] = []
        ids: list[str] = []

        for example in examples:
            ids.append(example.example_id)
            full_text = self._template.render(example.messages)
            prompt_messages = example.messages[:-1] if example.messages else ()
            prompt_text = self._template.render(prompt_messages) if prompt_messages else ""
            full_ids = self._encode_text(full_text)
            prompt_ids = self._encode_text(prompt_text) if config.mask_prompt else []
            labels = (
                apply_assistant_only_mask(full_ids, prompt_ids, ignore_index=config.ignore_index)
                if config.mask_prompt
                else list(full_ids)
            )
            full_seqs.append(full_ids)
            label_seqs.append(labels)

        if config.padding == "do_not_pad":
            # Rectangularize to max len in batch for TokenBatch invariant.
            width = max((len(s) for s in full_seqs), default=0)
            if config.truncation:
                width = min(width, config.max_length)
        elif config.padding == "longest":
            width = max((len(s) for s in full_seqs), default=0)
            if config.truncation:
                width = min(width, config.max_length)
        else:
            width = config.max_length

        padded_ids, masks = pad_sequences(
            full_seqs,
            max_length=width if width > 0 else 1,
            pad_id=self._pad_id,
            truncate=config.truncation,
        )
        padded_labels, _ = pad_sequences(
            label_seqs,
            max_length=width if width > 0 else 1,
            pad_id=config.ignore_index,
            truncate=config.truncation,
        )
        # Ensure padded label positions are ignore_index where attention is 0
        for row_i, mask in enumerate(masks):
            for col_i, m in enumerate(mask):
                if m == 0:
                    padded_labels[row_i][col_i] = config.ignore_index

        return TokenBatch(
            example_ids=tuple(ids),
            input_ids=tuple(tuple(r) for r in padded_ids),
            attention_mask=tuple(tuple(r) for r in masks),
            labels=tuple(tuple(r) for r in padded_labels),
            metadata=MappingProxyType({"tokenizer": self._name, "width": width}),
        )

    def _encode_text(self, text: str) -> list[int]:
        # Map each byte to 1..258; 0 reserved for pad.
        return [b + 1 for b in text.encode("utf-8")]

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            # Allow encode without explicit load in tests.
            if self._template is None:
                self._template = self._resolve_template(self._config.chat_template_name)
            self._loaded = True

    @staticmethod
    def _resolve_template(name: str) -> ChatTemplate:
        if not chat_template_registry.exists(name):
            raise DomainError(
                f"Chat template '{name}' is not registered. "
                f"Known: {', '.join(chat_template_registry.list()) or '(none)'}."
            )
        return chat_template_registry.get(name)()


class HuggingFaceTokenizerAdapter(TokenizerPort):
    """
    HuggingFace transformers AutoTokenizer backend.

    Requires the optional ``transformers`` dependency. Prefer
    DeterministicStubTokenizer for CI without model downloads.
    """

    def __init__(
        self,
        *,
        template: ChatTemplate | None = None,
        config: TokenizationConfig | None = None,
    ) -> None:
        self._template = template
        self._config = config or TokenizationConfig()
        self._tokenizer: object | None = None
        self._name = "huggingface"

    def load(self, model_ref: ModelRef) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DomainError(
                "transformers is required for HuggingFaceTokenizerAdapter. "
                "Install requirements/train.txt extras or use DeterministicStubTokenizer."
            ) from exc
        path = str(model_ref.local_path) if model_ref.local_path else model_ref.identifier
        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self._name = f"hf:{path}"
        if self._template is None:
            self._template = DeterministicStubTokenizer._resolve_template(
                self._config.chat_template_name
            )

    def vocabulary_size(self) -> int:
        if self._tokenizer is None:
            raise DomainError("Tokenizer is not loaded.")
        return int(self._tokenizer.vocab_size)  # type: ignore[attr-defined]

    def fingerprint(self) -> str:
        vocab = self.vocabulary_size() if self._tokenizer is not None else -1
        return fingerprint_tokenizer_identity(self._name, vocab_size=vocab)

    def encode_examples(self, examples: Sequence[TrainingExample]) -> TokenBatch:
        if self._tokenizer is None or self._template is None:
            raise DomainError("Call load() before encode_examples().")
        tok = self._tokenizer
        config = self._config
        full_seqs: list[list[int]] = []
        label_seqs: list[list[int]] = []
        ids: list[str] = []
        for example in examples:
            ids.append(example.example_id)
            full_text = self._template.render(example.messages)
            prompt_messages = example.messages[:-1] if example.messages else ()
            prompt_text = self._template.render(prompt_messages) if prompt_messages else ""
            full_ids = list(tok.encode(full_text, add_special_tokens=True))  # type: ignore[attr-defined]
            prompt_ids = (
                list(tok.encode(prompt_text, add_special_tokens=True))  # type: ignore[attr-defined]
                if config.mask_prompt and prompt_text
                else []
            )
            labels = (
                apply_assistant_only_mask(full_ids, prompt_ids, ignore_index=config.ignore_index)
                if config.mask_prompt
                else list(full_ids)
            )
            full_seqs.append(full_ids)
            label_seqs.append(labels)

        pad_id = int(getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0) or 0)
        width = (
            config.max_length
            if config.padding == "max_length"
            else max((len(s) for s in full_seqs), default=1)
        )
        if config.truncation:
            width = min(width, config.max_length)
        padded_ids, masks = pad_sequences(
            full_seqs, max_length=width, pad_id=pad_id, truncate=config.truncation
        )
        padded_labels, _ = pad_sequences(
            label_seqs,
            max_length=width,
            pad_id=config.ignore_index,
            truncate=config.truncation,
        )
        for row_i, mask in enumerate(masks):
            for col_i, m in enumerate(mask):
                if m == 0:
                    padded_labels[row_i][col_i] = config.ignore_index
        return TokenBatch(
            example_ids=tuple(ids),
            input_ids=tuple(tuple(r) for r in padded_ids),
            attention_mask=tuple(tuple(r) for r in masks),
            labels=tuple(tuple(r) for r in padded_labels),
            metadata=MappingProxyType({"tokenizer": self._name}),
        )


def register_default_tokenizers(*, overwrite: bool = False) -> None:
    """Register tokenizer backends."""
    mapping = {
        "stub": DeterministicStubTokenizer,
        "huggingface": HuggingFaceTokenizerAdapter,
    }
    for key, cls in mapping.items():
        if tokenizer_registry.exists(key) and not overwrite:
            continue
        tokenizer_registry.register(key, cls, overwrite=overwrite)
