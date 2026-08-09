"""Canonical Training System Contract (TR-2).

Frozen AIODOO System WHAT contracts → versioned Training record formats.

This package does **not** redefine System behavior. It teaches behavior.
Provider/adapter capabilities (``aiodoo_contract``) remain a separate plane —
see :mod:`aiodoo_training.contract`.

System source of truth: ``aiodoo-core`` (ExecutionWorkUnit, capabilities, …).
Constants here are synchronized copies so Training stays independent of Core
at import time (ECO-1). Optional sync tests may compare against Core when
present on PYTHONPATH.
"""

from __future__ import annotations

from aiodoo_training.system_training_contract.forbidden import (
    FORBIDDEN_ARG_KEYS,
    FORBIDDEN_BACKEND_ACTIONS,
    FORBIDDEN_IMPL_IDS,
    assert_no_forbidden_how,
)
from aiodoo_training.system_training_contract.projection import (
    ProjectionResult,
    ProjectionStatus,
    Provenance,
    project_historical_record,
)
from aiodoo_training.system_training_contract.records import (
    RECORD_TYPES,
    CapabilityIntentRecord,
    EngineeringFeedbackRecord,
    EngineeringStateRecord,
    EvaluationJudgmentRecord,
    LoopDecisionRecord,
    ObservationRecord,
    PlanningDecisionRecord,
    TrainingRecordBase,
    WorkUnitRecord,
    DecisionContextRecord,
    validate_record_mapping,
)
from aiodoo_training.system_training_contract.taxonomy import (
    DEVELOPMENT_PROVIDER_CAPABILITIES,
    ENGINEERING_CAPABILITY_IDS,
    PREFERRED_ENGINEERING_CAPABILITY_IDS,
    PROVIDER_CAPABILITY_IDS,
    REASONING_PROVIDER_CAPABILITIES,
    TaxonomyPlane,
    classify_capability_id,
)
from aiodoo_training.system_training_contract.version import (
    SYSTEM_TRAINING_CONTRACT_VERSION,
)

__all__ = [
    "SYSTEM_TRAINING_CONTRACT_VERSION",
    "TaxonomyPlane",
    "ENGINEERING_CAPABILITY_IDS",
    "PREFERRED_ENGINEERING_CAPABILITY_IDS",
    "PROVIDER_CAPABILITY_IDS",
    "DEVELOPMENT_PROVIDER_CAPABILITIES",
    "REASONING_PROVIDER_CAPABILITIES",
    "classify_capability_id",
    "FORBIDDEN_IMPL_IDS",
    "FORBIDDEN_BACKEND_ACTIONS",
    "FORBIDDEN_ARG_KEYS",
    "assert_no_forbidden_how",
    "RECORD_TYPES",
    "TrainingRecordBase",
    "CapabilityIntentRecord",
    "WorkUnitRecord",
    "PlanningDecisionRecord",
    "ObservationRecord",
    "EngineeringFeedbackRecord",
    "EngineeringStateRecord",
    "DecisionContextRecord",
    "LoopDecisionRecord",
    "EvaluationJudgmentRecord",
    "validate_record_mapping",
    "Provenance",
    "ProjectionStatus",
    "ProjectionResult",
    "project_historical_record",
]
