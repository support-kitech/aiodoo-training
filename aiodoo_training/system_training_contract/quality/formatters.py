"""FP2-native Training pack formatters (TR-4).

Canonical Training Record → TrainingExample without Protocol V1 / aiodoo_contract
projection. Does not train adapters.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from aiodoo_training.datasets.formatters.base import user_assistant
from aiodoo_training.datasets.mixing import stable_example_id
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.examples import TrainingExample, freeze_messages
from aiodoo_training.system_training_contract.records import TrainingRecordError, validate_record_mapping
from aiodoo_training.system_training_contract.taxonomy import (
    DEVELOPMENT_PROVIDER_CAPABILITIES,
    REASONING_PROVIDER_CAPABILITIES,
)

_RECORD_TO_DEFAULT_DATASET: dict[str, DatasetType] = {
    "capability_intent": DatasetType.PLANNER,
    "execution_work_unit": DatasetType.EXECUTION,
    "planning_decision": DatasetType.PLANNER,
    "observation": DatasetType.EXECUTION,
    "engineering_feedback": DatasetType.PLANNER,
    "engineering_state": DatasetType.PLANNER,
    "decision_context": DatasetType.PLANNER,
    "loop_decision": DatasetType.PLANNER,
    "evaluation_judgment": DatasetType.EVALUATION,
}


def _dataset_for_record(record: Mapping[str, Any]) -> DatasetType:
    provider = str(record.get("provider_capability") or "").strip()
    if provider:
        try:
            return DatasetType(provider)
        except ValueError:
            pass
    rtype = str(record.get("record_type") or "")
    return _RECORD_TO_DEFAULT_DATASET.get(rtype, DatasetType.MIXED)


def _instruction_for(record: Mapping[str, Any]) -> str:
    rtype = str(record.get("record_type") or "")
    system = str(record.get("system_contract") or "")
    return (
        f"AIODOO System Training Contract v1.0.0 — emit model-facing WHAT only.\n"
        f"record_type={rtype}\n"
        f"system_contract={system}\n"
        f"Never emit local_* implementations, backends, shell/git/pytest commands, "
        f"or provider pack IDs as Engineering actions."
    )


def format_fp2_record(record: Mapping[str, Any], *, index: int = 0) -> TrainingExample:
    """Format one canonical FP2 record into a TrainingExample."""
    validated = validate_record_mapping(record)
    dataset_type = _dataset_for_record(validated)
    user_payload = {
        "input": validated.get("input"),
        "evidence": validated.get("evidence"),
        "provider_capability": validated.get("provider_capability"),
        "domain_specialization": validated.get("domain_specialization"),
    }
    # Prefer expected_output as label; for evidence-only families, echo structured evidence task
    assistant_payload = validated.get("expected_output")
    if not assistant_payload:
        assistant_payload = {
            "record_type": validated.get("record_type"),
            "evidence": validated.get("evidence"),
        }
    user_text = _instruction_for(validated) + "\n\n" + json.dumps(
        user_payload, sort_keys=True, ensure_ascii=False
    )
    assistant_text = json.dumps(assistant_payload, sort_keys=True, ensure_ascii=False)
    meta = dict(validated.get("metadata") or {})
    meta.update(
        {
            "fp2_native": True,
            "training_contract_version": validated.get("training_contract_version"),
            "record_type": validated.get("record_type"),
            "record_id": validated.get("record_id"),
            "adapter_plane": (
                "development"
                if str(validated.get("provider_capability") or "")
                in DEVELOPMENT_PROVIDER_CAPABILITIES
                else "reasoning"
                if str(validated.get("provider_capability") or "")
                in REASONING_PROVIDER_CAPABILITIES
                else "unknown"
            ),
        }
    )
    messages = freeze_messages(
        (
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        )
    )
    return TrainingExample(
        example_id=stable_example_id(dataset_type.value, validated, index),
        dataset_type=dataset_type,
        messages=messages,
        metadata=MappingProxyType(meta),
    )


def format_fp2_pack(
    records: list[Mapping[str, Any]],
    *,
    pack: str,
) -> list[TrainingExample]:
    """Filter records for a Development or Reasoning pack and format them."""
    # Lazy import avoids generators.__init__ ↔ formatters circular import.
    from aiodoo_training.system_training_contract.generators.mapping import (
        DEVELOPMENT_RECORD_TYPES,
        REASONING_RECORD_TYPES,
    )

    pack = pack.strip().lower()
    if pack not in {"development", "reasoning"}:
        raise ValueError("pack must be 'development' or 'reasoning'")
    allowed_types = DEVELOPMENT_RECORD_TYPES if pack == "development" else REASONING_RECORD_TYPES
    allowed_providers = (
        DEVELOPMENT_PROVIDER_CAPABILITIES if pack == "development" else REASONING_PROVIDER_CAPABILITIES
    )
    out: list[TrainingExample] = []
    for i, raw in enumerate(records):
        rtype = str(raw.get("record_type") or "")
        provider = str(raw.get("provider_capability") or "")
        if rtype not in allowed_types:
            continue
        if provider and provider not in allowed_providers:
            # Shared record types may be labeled with the other plane's provider;
            # include when record type is allowed for this pack.
            if rtype not in allowed_types:
                continue
        try:
            out.append(format_fp2_record(raw, index=i))
        except TrainingRecordError:
            continue
    return out
