"""Chat template implementations (family-specific rendering)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from aiodoo_training.ports.chat_template import ChatTemplate
from aiodoo_training.registries import chat_template_registry


class SimpleRoleChatTemplate(ChatTemplate):
    """
    Deterministic role-tag template used as a portable default.

    Format:
        <|role|>\\n{content}\\n
    """

    def __init__(self, name: str, family: str, *, version: str = "1") -> None:
        self._name = name
        self._family = family
        self._version = version

    @property
    def name(self) -> str:
        return self._name

    @property
    def family(self) -> str:
        return self._family

    def render(self, messages: Sequence[Mapping[str, str]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip() or "user"
            content = str(message.get("content", ""))
            parts.append(f"<|{role}|>\n{content}\n")
        return "".join(parts).rstrip() + "\n"

    def fingerprint(self) -> str:
        material = f"{self._name}:{self._family}:simple_role:v{self._version}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


class QwenChatTemplate(SimpleRoleChatTemplate):
    """Qwen-family chat template (Phase 1 portable renderer)."""

    def __init__(self) -> None:
        super().__init__("qwen", "qwen", version="1")


class LlamaChatTemplate(SimpleRoleChatTemplate):
    """Llama-family placeholder template (same renderer, distinct fingerprint)."""

    def __init__(self) -> None:
        super().__init__("llama", "llama", version="1")


class MistralChatTemplate(SimpleRoleChatTemplate):
    def __init__(self) -> None:
        super().__init__("mistral", "mistral", version="1")


def register_default_chat_templates(*, overwrite: bool = False) -> None:
    """Register built-in chat templates into chat_template_registry."""
    defaults: dict[str, type[ChatTemplate]] = {
        "qwen": QwenChatTemplate,
        "llama": LlamaChatTemplate,
        "mistral": MistralChatTemplate,
    }
    for key, cls in defaults.items():
        if chat_template_registry.exists(key) and not overwrite:
            continue
        chat_template_registry.register(key, cls, overwrite=overwrite)
