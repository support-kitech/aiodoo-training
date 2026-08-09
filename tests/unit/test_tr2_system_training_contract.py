"""TR-2 — Canonical Training System Contract unit tests (tiny fixtures only)."""

from __future__ import annotations

import pytest

from aiodoo_training.system_training_contract import (
    SYSTEM_TRAINING_CONTRACT_VERSION,
    CapabilityIntentRecord,
    DecisionContextRecord,
    EngineeringFeedbackRecord,
    EngineeringStateRecord,
    LoopDecisionRecord,
    ObservationRecord,
    PlanningDecisionRecord,
    ProjectionStatus,
    TaxonomyPlane,
    WorkUnitRecord,
    assert_no_forbidden_how,
    classify_capability_id,
    project_historical_record,
    validate_record_mapping,
)
from aiodoo_training.system_training_contract.forbidden import ForbiddenHowError
from aiodoo_training.system_training_contract.records import TrainingRecordError
from aiodoo_training.system_training_contract.taxonomy import (
    DEVELOPMENT_PROVIDER_CAPABILITIES,
    PREFERRED_ENGINEERING_CAPABILITY_IDS,
    PROVIDER_CAPABILITY_IDS,
    REASONING_PROVIDER_CAPABILITIES,
)


def test_contract_version_is_explicit() -> None:
    assert SYSTEM_TRAINING_CONTRACT_VERSION == "1.0.0"


def test_provider_vs_engineering_taxonomy_separated() -> None:
    assert DEVELOPMENT_PROVIDER_CAPABILITIES.isdisjoint(PREFERRED_ENGINEERING_CAPABILITY_IDS)
    assert REASONING_PROVIDER_CAPABILITIES.isdisjoint(PREFERRED_ENGINEERING_CAPABILITY_IDS)
    assert classify_capability_id("coding") is TaxonomyPlane.PROVIDER
    assert classify_capability_id("planner") is TaxonomyPlane.PROVIDER
    assert classify_capability_id("workspace.write") is TaxonomyPlane.ENGINEERING
    assert classify_capability_id("validation.run") is TaxonomyPlane.ENGINEERING
    assert classify_capability_id("local_workspace") is TaxonomyPlane.FORBIDDEN_HOW
    assert "coding" in PROVIDER_CAPABILITY_IDS
    assert "workspace.write" not in PROVIDER_CAPABILITY_IDS


def test_forbidden_how_rejection() -> None:
    with pytest.raises(ForbiddenHowError):
        assert_no_forbidden_how(capability_id="local_git")
    with pytest.raises(ForbiddenHowError):
        assert_no_forbidden_how(args={"command": "ls"})
    with pytest.raises(ForbiddenHowError):
        assert_no_forbidden_how(args={"implementation_id": "local_program"})


def test_capability_intent_valid() -> None:
    rec = CapabilityIntentRecord(
        record_type="capability_intent",
        record_id="ci-1",
        capability_id="workspace.write",
        objective="Create module file",
        args={"path": "models/partner.py"},
        provider_capability="planner",
    )
    data = rec.to_dict()
    assert data["expected_output"]["capability_id"] == "workspace.write"
    assert validate_record_mapping(data)["record_id"] == "ci-1"


def test_capability_intent_rejects_provider_pack_as_engineering() -> None:
    with pytest.raises(TrainingRecordError):
        CapabilityIntentRecord(
            record_type="capability_intent",
            record_id="bad",
            capability_id="coding",
            objective="x",
        ).validate()


def test_capability_intent_rejects_transitional_shell() -> None:
    with pytest.raises(TrainingRecordError):
        CapabilityIntentRecord(
            record_type="capability_intent",
            record_id="bad",
            capability_id="shell",
            objective="run something",
        ).validate()


def test_work_unit_valid() -> None:
    rec = WorkUnitRecord(
        record_type="execution_work_unit",
        record_id="wu-1",
        work_id="ewu-fixture-1",
        capability_id="execution.execute_program",
        objective="Run entrypoint",
        expected_outputs={"exit_code": 0},
        provider_capability="execution",
    )
    assert rec.to_dict()["expected_output"]["work_id"] == "ewu-fixture-1"


def test_planning_decision_and_loop_kinds() -> None:
    plan = PlanningDecisionRecord(
        record_type="planning_decision",
        record_id="pd-1",
        goal="Add model field",
        decision_kind="replan",
        summary="Write then validate",
        steps=(
            {"action": "workspace.write", "args": {"path": "a.py"}},
            {"action": "validation.run", "args": {}},
        ),
        provider_capability="planner",
    )
    assert len(plan.to_dict()["expected_output"]["steps"]) == 2

    complete = PlanningDecisionRecord(
        record_type="planning_decision",
        record_id="pd-2",
        goal="Done",
        decision_kind="complete",
        steps=(),
        provider_capability="planner",
    )
    complete.validate()

    with pytest.raises(TrainingRecordError):
        PlanningDecisionRecord(
            record_type="planning_decision",
            record_id="pd-bad",
            goal="x",
            decision_kind="complete",
            steps=({"action": "workspace.read", "args": {}},),
        ).validate()

    for kind in ("replan", "complete", "escalate"):
        LoopDecisionRecord(
            record_type="loop_decision",
            record_id=f"ld-{kind}",
            decision_kind=kind,
            reason=f"because {kind}",
            provider_capability="planner",
        ).validate()


def test_observation_feedback_state_decision_context() -> None:
    ObservationRecord(
        record_type="observation",
        record_id="obs-1",
        kind="execution_result",
        status="succeeded",
        capability_id="workspace.write",
        summary="wrote file",
    ).validate()

    EngineeringFeedbackRecord(
        record_type="engineering_feedback",
        record_id="fb-1",
        objective="ship change",
        objective_state="partial",
        continuation_options=("replan", "complete"),
    ).validate()

    EngineeringStateRecord(
        record_type="engineering_state",
        record_id="st-1",
        objective="ship change",
        cycle_index=2,
    ).validate()

    DecisionContextRecord(
        record_type="decision_context",
        record_id="dc-1",
        objective="ship change",
        missing_outcomes=("validation",),
        blockers=(),
        continuation_hint="replan",
    ).validate()


def test_projection_planner_unsupported() -> None:
    result = project_historical_record(
        {"steps": [{"action": "create_file", "path": "x.py"}]},
        source_dataset="planner_odoo_v1",
        source_record_id="p-1",
    )
    assert result.status is ProjectionStatus.UNSUPPORTED
    assert result.provenance.source_record_id == "p-1"
    assert "create_file" in result.reasons[0]


def test_projection_execution_lossy_unsupported() -> None:
    result = project_historical_record(
        {"op": "apply_artifact", "artifact_id": "a1"},
        source_dataset="execution_train",
        source_record_id="e-1",
    )
    assert result.status is ProjectionStatus.UNSUPPORTED
    assert result.provider_capability == "execution"


def test_projection_coding_partial_not_work_unit() -> None:
    result = project_historical_record(
        {"objective": "implement helper", "code": "def f(): ..."},
        source_dataset="coding_sft",
        source_record_id="c-1",
    )
    assert result.status is ProjectionStatus.PARTIALLY_PROJECTED
    assert result.provider_capability == "coding"
    assert result.canonical is not None
    assert result.canonical.get("projection_envelope") is True
    assert result.canonical.get("metadata", {}).get("do_not_auto_convert_to_work_unit") is True


def test_projection_repair_explicit_engineering() -> None:
    result = project_historical_record(
        {
            "capability_id": "execution.repair",
            "objective": "fix import",
            "args": {"path": "m.py"},
        },
        source_dataset="repair_sft",
        source_record_id="r-1",
    )
    assert result.status is ProjectionStatus.PROJECTED
    assert result.canonical is not None
    assert result.canonical["expected_output"]["capability_id"] == "execution.repair"


def test_projection_forbidden_how_rejected() -> None:
    result = project_historical_record(
        {"capability_id": "local_workspace"},
        source_dataset="misc",
        source_record_id="x-1",
    )
    assert result.status is ProjectionStatus.REJECTED


def test_projection_approval_to_loop_decision() -> None:
    result = project_historical_record(
        {"decision_kind": "approve", "reason": "looks good"},
        source_dataset="approval_sft",
        source_record_id="a-1",
    )
    assert result.status is ProjectionStatus.PROJECTED
    assert result.canonical is not None
    assert result.canonical["record_type"] == "loop_decision"


def test_odoo_specialization_preserved() -> None:
    result = project_historical_record(
        {"objective": "add odoo field", "domain": "odoo"},
        source_dataset="coding_odoo",
        source_record_id="o-1",
    )
    assert result.domain_specialization == "odoo"
    assert result.status is ProjectionStatus.PARTIALLY_PROJECTED


def test_provenance_preserved_on_unsupported() -> None:
    result = project_historical_record(
        {"foo": "bar"},
        source_dataset="unknown_pack",
        source_record_id="u-9",
        source_schema_version="protocol-v1",
    )
    assert result.status is ProjectionStatus.UNSUPPORTED
    assert result.provenance.source_dataset == "unknown_pack"
    assert result.provenance.source_schema_version == "protocol-v1"
    assert result.provenance.projection_version == "1.0.0"
