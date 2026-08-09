"""FP2-native corpus generators (TR-3).

Produce canonical ``SYSTEM_TRAINING_CONTRACT_VERSION`` records.
Fixture-first: tiny deterministic corpora only — no mass regeneration.
"""

from __future__ import annotations

from aiodoo_training.system_training_contract.generators.emit import (
    DATASET_GENERATION_VERSION,
    GENERATOR_NAMES,
    emit_all_fixtures,
    generate_all,
)
from aiodoo_training.system_training_contract.generators.mapping import (
    CONTEXT_ALLOWED_RECORD_TYPES,
    CONTEXT_REJECTED_RECORD_TYPES,
    DEVELOPMENT_RECORD_TYPES,
    EVALUATION_ALLOWED_RECORD_TYPES,
    EVALUATION_REJECTED_RECORD_TYPES,
    REASONING_RECORD_TYPES,
    record_provider_capabilities,
)
from aiodoo_training.system_training_contract.generators.controlled_batch import (
    CONTROLLED_BATCH_VERSION,
    emit_controlled_batch,
    generate_controlled_batch_records,
)
from aiodoo_training.system_training_contract.generators.context import (
    CONTEXT_CORPUS_VERSION,
    emit_context_fixtures,
    generate_context_records,
)
from aiodoo_training.system_training_contract.generators.context_controlled import (
    CONTEXT_CONTROLLED_VERSION,
    analyze_context_controlled,
    emit_context_controlled_corpus,
    generate_context_controlled_records,
)
from aiodoo_training.system_training_contract.generators.conversation_controlled import (
    generate_conversation_controlled_records,
)
from aiodoo_training.system_training_contract.generators.approval_controlled import (
    generate_approval_controlled_records,
)
from aiodoo_training.system_training_contract.generators.evaluation_semantics import (
    EVALUATION_CONTRACT_DECISION,
    EVALUATION_SEMANTIC_DEFINITION,
)
from aiodoo_training.system_training_contract.generators.evaluation_controlled import (
    analyze_evaluation_controlled,
    emit_evaluation_controlled_corpus,
    generate_evaluation_controlled_records,
)
from aiodoo_training.system_training_contract.generators.reasoning_sparse_emit import (
    REASONING_SPARSE_VERSION,
    analyze_approval_controlled,
    analyze_conversation_controlled,
    emit_reasoning_sparse_corpora,
)

__all__ = [
    "DATASET_GENERATION_VERSION",
    "GENERATOR_NAMES",
    "DEVELOPMENT_RECORD_TYPES",
    "REASONING_RECORD_TYPES",
    "CONTEXT_ALLOWED_RECORD_TYPES",
    "CONTEXT_REJECTED_RECORD_TYPES",
    "EVALUATION_ALLOWED_RECORD_TYPES",
    "EVALUATION_REJECTED_RECORD_TYPES",
    "CONTEXT_CORPUS_VERSION",
    "CONTEXT_CONTROLLED_VERSION",
    "REASONING_SPARSE_VERSION",
    "EVALUATION_SEMANTIC_DEFINITION",
    "EVALUATION_CONTRACT_DECISION",
    "record_provider_capabilities",
    "generate_all",
    "emit_all_fixtures",
    "generate_context_records",
    "emit_context_fixtures",
    "generate_context_controlled_records",
    "emit_context_controlled_corpus",
    "analyze_context_controlled",
    "generate_conversation_controlled_records",
    "generate_approval_controlled_records",
    "generate_evaluation_controlled_records",
    "analyze_evaluation_controlled",
    "emit_evaluation_controlled_corpus",
    "analyze_conversation_controlled",
    "analyze_approval_controlled",
    "emit_reasoning_sparse_corpora",
    "CONTROLLED_BATCH_VERSION",
    "generate_controlled_batch_records",
    "emit_controlled_batch",
]
