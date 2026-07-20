"""Bridge between aiodoo-training's dataset records and ``aiodoo_contract``.

This package is training's consumer-side equivalent of aiodoo-datasets'
``generators/common/contract`` package: it owns nothing that
``aiodoo_contract`` already defines (no schemas, no prompt builder, no chat
templates, no validators) — it only *projects* a raw protocol JSONL record
(as produced by aiodoo-datasets and read by
:class:`aiodoo_training.datasets.reader.ProtocolRecordReader`) onto the
canonical ``aiodoo_contract.schemas`` request/response shape for its
capability, and *bridges* that projection into a rendered prompt via
``aiodoo_contract.prompts.CapabilityPromptBuilder``.

Why this logic exists here and not just in aiodoo-datasets: aiodoo-training
consumes datasets as on-disk JSONL, not as a Python import of
aiodoo-datasets internals (see ``docs/adr/0008-dependency-graph.md`` in
aiodoo-contract — there is no training → datasets Python dependency edge).
The record *shape* is the data contract between the two repositories, so
each consumer that needs to interpret it independently re-implements this
narrow projection step against the one canonical target
(:mod:`aiodoo_contract.schemas`) — it does not redefine the target itself.
"""

from aiodoo_training.contract.adapters import (
    SUPPORTED_CAPABILITIES,
    ContractAdapterError,
    ContractProjection,
    project_record,
)
from aiodoo_training.contract.prompt_bridge import (
    build_training_example,
    render_capability_prompt,
    serialize_response,
)
from aiodoo_training.contract.version_check import (
    TRAINING_CONTRACT_VERSION,
    ContractVersionError,
    ensure_contract_compatible,
)

__all__ = [
    "SUPPORTED_CAPABILITIES",
    "TRAINING_CONTRACT_VERSION",
    "ContractAdapterError",
    "ContractProjection",
    "ContractVersionError",
    "build_training_example",
    "ensure_contract_compatible",
    "project_record",
    "render_capability_prompt",
    "serialize_response",
]
