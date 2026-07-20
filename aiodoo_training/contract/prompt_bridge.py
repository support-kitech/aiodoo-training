"""Bridge a contract projection into a training-ready prompt + label.

This is the only place in aiodoo-training that turns a capability request
into prompt text — it delegates every formatting decision to
``aiodoo_contract.prompts.CapabilityPromptBuilder`` (ADR-0003). Training
must not hand-assemble instruction/context strings itself; see
``docs/adr/0003-prompt-builder-ownership.md`` in aiodoo-contract.

The assistant/label side is the other half of the contract-consistency
fix this module exists for: the text a training example teaches the model
to produce is the canonical ``CapabilityResponse`` JSON — the exact shape
a runtime parser (:mod:`aiodoo_contract.parsers`) will later decode —
rather than the richer, training-pedagogy-shaped ``record["output"]`` blob
the dataset generator produced it from. Teaching the raw dataset blob
instead of the contract response is precisely the "training teaches one
schema, runtime validates another" defect class this contract exists to
close.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any

from aiodoo_contract.prompts import CapabilityPromptBuilder, RenderedPrompt
from aiodoo_contract.schemas.base import CapabilityResponse

from aiodoo_training.contract.adapters import ContractProjection, project_record
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.examples import TrainingExample, freeze_messages

__all__ = [
    "build_training_example",
    "render_capability_prompt",
    "serialize_response",
]

_PROMPT_BUILDER = CapabilityPromptBuilder()


def render_capability_prompt(projection: ContractProjection) -> RenderedPrompt:
    """Render ``projection.request`` via the canonical Capability Prompt Builder.

    This is the single call site every capability's prompt construction
    goes through — see :class:`aiodoo_contract.prompts.CapabilityPromptBuilder`.
    """
    return _PROMPT_BUILDER.build_from_request(projection.request)


def serialize_response(response: CapabilityResponse) -> str:
    """Serialize ``response`` to the canonical JSON text a model is trained to produce.

    Deterministic (``sort_keys=True``) so example ids and any downstream
    fingerprinting stay stable across runs. The output is exactly what
    :class:`aiodoo_contract.parsers.capability.CapabilityParser` expects to
    parse back — see ``tests/contract/test_prompt_bridge.py`` for the
    round-trip proof.
    """
    payload = response.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_training_example(
    *,
    dataset_type: DatasetType,
    capability: str,
    record: dict[str, Any],
    example_id: str,
) -> TrainingExample:
    """Project ``record`` onto its contract shape and build a `TrainingExample`.

    Raises:
        aiodoo_training.contract.adapters.ContractAdapterError: if ``record``
            cannot be projected onto ``capability``'s contract shape.
    """
    projection = project_record(capability, record)
    prompt = render_capability_prompt(projection)
    label = serialize_response(projection.response)

    turns: list[dict[str, str]] = []
    if prompt.system:
        turns.append({"role": "system", "content": prompt.system})
    turns.append({"role": "user", "content": prompt.user})
    turns.append({"role": "assistant", "content": label})

    messages = freeze_messages(turns)
    raw_metadata = record.get("metadata", {})
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {"raw_metadata": raw_metadata}
    metadata = {
        **metadata,
        "contract_version": projection.request.contract_version,
        "capability": capability,
    }
    return TrainingExample(
        example_id=example_id,
        dataset_type=dataset_type,
        messages=messages,
        metadata=MappingProxyType(metadata),
    )
