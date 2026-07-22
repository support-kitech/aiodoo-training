"""Chat template implementations — thin adapters onto `aiodoo_contract.templates`.

Training never formats prompts independently (ADR-0004 in aiodoo-contract):
every class here renders by delegating to the canonical
:mod:`aiodoo_contract.templates` implementation for its family, converting
between the frozen :class:`~aiodoo_training.ports.chat_template.ChatTemplate`
port shape (``render(messages) -> str``, dict messages) this repository's
tokenizer/pipeline call sites depend on, and the contract's
:class:`~aiodoo_contract.templates.base.BaseChatTemplate` shape
(``render_conversation``/``render_generation_prompt``, `ChatMessage` objects).

``ChatTemplate.render(messages)`` is always called with either a full closed
conversation (system/user/assistant turns already including the label) for
training-time tokenization, or that conversation minus its final turn for
prompt-only masking — both are "no open trailing turn" renderings, i.e.
:meth:`~aiodoo_contract.templates.base.BaseChatTemplate.render_conversation`,
never :meth:`render_generation_prompt`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from aiodoo_contract.templates import BaseChatTemplate, get_chat_template
from aiodoo_contract.templates.messages import ChatMessage, ChatRole
from aiodoo_contract.version import CONTRACT_VERSION

from aiodoo_training.ports.chat_template import ChatTemplate
from aiodoo_training.registries import chat_template_registry

_ROLE_BY_VALUE = {role.value: role for role in ChatRole}


def _to_chat_messages(messages: Sequence[Mapping[str, str]]) -> tuple[ChatMessage, ...]:
    converted: list[ChatMessage] = []
    for message in messages:
        role_raw = str(message.get("role", "user")).strip().lower() or "user"
        role = _ROLE_BY_VALUE.get(role_raw, ChatRole.USER)
        converted.append(ChatMessage(role=role, content=str(message.get("content", ""))))
    return tuple(converted)


class ContractBackedChatTemplate(ChatTemplate):
    """Adapts one `aiodoo_contract.templates` family onto the `ChatTemplate` port.

    Args:
        name: the registry key/template identity in aiodoo-training
            (e.g. ``"qwen"``).
        family: the model family this template targets.
        contract_template_name: the name to resolve via
            :func:`aiodoo_contract.templates.get_chat_template`. Defaults to
            ``name`` — pass a different value to intentionally fall back to
            a contract template for a family the contract has no dedicated
            rendering for (e.g. ``llama``/``mistral`` -> ``generic``).
    """

    def __init__(
        self, name: str, family: str, *, contract_template_name: str | None = None
    ) -> None:
        self._name = name
        self._family = family
        self._contract_template_name = contract_template_name or name
        self._contract_template: BaseChatTemplate = get_chat_template(self._contract_template_name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def family(self) -> str:
        return self._family

    def render(self, messages: Sequence[Mapping[str, str]]) -> str:
        return self._contract_template.render_conversation(_to_chat_messages(messages))

    def fingerprint(self) -> str:
        # Identity depends on the *contract* template actually used plus the
        # installed contract version, so a contract upgrade that changes
        # rendering invalidates any tokenization cache keyed on this.
        material = (
            f"{self._name}:{self._family}:aiodoo_contract:"
            f"{self._contract_template_name}:v{CONTRACT_VERSION}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


class QwenChatTemplate(ContractBackedChatTemplate):
    """Qwen-family chat template — delegates to `aiodoo_contract.templates.QwenChatTemplate`."""

    def __init__(self) -> None:
        super().__init__("qwen", "qwen")


class DeepSeekChatTemplate(ContractBackedChatTemplate):
    """DeepSeek-family chat template — delegates to the contract's DeepSeek renderer."""

    def __init__(self) -> None:
        super().__init__("deepseek", "deepseek")


class GenericChatTemplate(ContractBackedChatTemplate):
    """Plain-text fallback template — delegates to the contract's generic renderer."""

    def __init__(self) -> None:
        super().__init__("generic", "generic")


class LlamaChatTemplate(ContractBackedChatTemplate):
    """Llama-family template.

    aiodoo_contract has no dedicated Llama rendering yet (only qwen /
    deepseek / generic — see ``aiodoo_contract/templates/registry.py``), so
    this intentionally falls back to the contract's generic template rather
    than inventing a training-local Llama format. Documented in
    ``CONTRACT_ADOPTION.md``.
    """

    def __init__(self) -> None:
        super().__init__("llama", "llama", contract_template_name="generic")


class MistralChatTemplate(ContractBackedChatTemplate):
    """Mistral-family template — falls back to the contract's generic template (see Llama)."""

    def __init__(self) -> None:
        super().__init__("mistral", "mistral", contract_template_name="generic")


def register_default_chat_templates(*, overwrite: bool = False) -> None:
    """Register built-in chat templates into chat_template_registry."""
    defaults: dict[str, type[ChatTemplate]] = {
        "qwen": QwenChatTemplate,
        "deepseek": DeepSeekChatTemplate,
        "generic": GenericChatTemplate,
        "llama": LlamaChatTemplate,
        "mistral": MistralChatTemplate,
    }
    for key, cls in defaults.items():
        if chat_template_registry.exists(key) and not overwrite:
            continue
        chat_template_registry.register(key, cls, overwrite=overwrite)
