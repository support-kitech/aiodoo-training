"""TR-7 continuity expansion — meaningful multi-cycle Reasoning scenarios."""

from __future__ import annotations

from typing import Any

from aiodoo_training.system_training_contract.generators.common import fixture_metadata
from aiodoo_training.system_training_contract.records import (
    DecisionContextRecord,
    EngineeringStateRecord,
    LoopDecisionRecord,
)

TR7_BATCH_VERSION = "fp2-controlled-2.0.0-tr7"


def _rid(kind: str, n: int) -> str:
    return f"fp2-tr7-{kind}-{n:04d}"


# Each scenario defines one continuity triple (state + decision_context + loop).
# Families are distinct; no automatic failure→repair→validation pipeline.
_SCENARIOS: tuple[dict[str, Any], ...] = (
    # 1 success → COMPLETE
    {
        "family": "cont_success_complete",
        "objective": "Ship inspected repository change",
        "cycle": 2,
        "objective_state": "complete",
        "execution_state": "succeeded",
        "validation_status": "passed",
        "missing": (),
        "blockers": (),
        "hint": "complete",
        "loop": "complete",
        "reason": "Objective complete; validation passed; no blockers",
        "history": ({"cycle_index": 1, "objective_state": "in_progress", "historical": True},),
        "domain": None,
    },
    # 2 operation success → objective incomplete → REPLAN
    {
        "family": "cont_op_ok_obj_incomplete_replan",
        "objective": "Add res.partner field via models/partner.py and close objective",
        "cycle": 1,
        "objective_state": "incomplete",
        "execution_state": "succeeded",
        "validation_status": "pending",
        "missing": ("validation",),
        "blockers": (),
        "hint": "replan",
        "loop": "replan",
        "reason": "Write to models/partner.py succeeded but Odoo objective incomplete — missing validation",
        "history": (),
        "domain": "odoo",
        "current_fields": {
            "last_capability": "workspace.write",
            "operation_ok": True,
            "path": "models/partner.py",
        },
    },
    # 3 failure → evidence → REPLAN
    {
        "family": "cont_fail_replan",
        "objective": "Publish change artifact",
        "cycle": 1,
        "objective_state": "incomplete",
        "execution_state": "failed",
        "validation_status": "not_run",
        "missing": ("artifact",),
        "failures": ("publish_failed",),
        "blockers": (),
        "hint": "replan",
        "loop": "replan",
        "reason": "Publish failed; choose next WHAT from evidence — not a forced repair",
        "history": (),
        "domain": None,
    },
    # 4 failure → evidence → ESCALATE
    {
        "family": "cont_fail_escalate",
        "objective": "Reach remote health endpoint",
        "cycle": 2,
        "objective_state": "escalated",
        "execution_state": "failed",
        "validation_status": "not_run",
        "missing": (),
        "failures": ("http_unreachable",),
        "blockers": ("remote_unreachable",),
        "hint": "escalate",
        "loop": "escalate",
        "reason": "Retries exhausted; remote dependency unreachable — escalate",
        "history": ({"cycle_index": 1, "objective_state": "incomplete", "historical": True},),
        "domain": None,
    },
    # 5 failure → repair → new evidence → continuation (not auto-validate)
    {
        "family": "cont_fail_repair_continue",
        "objective": "Restore importable Odoo module under addons/custom_partner",
        "cycle": 2,
        "objective_state": "in_progress",
        "execution_state": "succeeded",
        "validation_status": "pending",
        "repair_status": "applied",
        "missing": ("validation",),
        "blockers": (),
        "hint": "continue",
        "loop": "continue",
        "reason": "Repair of addons/custom_partner observed; objective still incomplete — continue",
        "history": (
            {
                "cycle_index": 1,
                "objective_state": "failed",
                "note": "import_error",
                "historical": True,
            },
        ),
        "domain": "odoo",
        "current_fields": {"path": "addons/custom_partner/__manifest__.py"},
    },
    # 6 repair → validation evidence → still incomplete
    {
        "family": "cont_repair_validate_still_incomplete",
        "objective": "Ship models/partner.py field change for res.partner",
        "cycle": 3,
        "objective_state": "incomplete",
        "execution_state": "partial",
        "validation_status": "failed",
        "repair_status": "applied",
        "missing": ("validation",),
        "failures": ("validation_failed",),
        "blockers": (),
        "hint": "replan",
        "loop": "replan",
        "reason": "Odoo validation still failing after repair — replan next WHAT from evidence",
        "history": (
            {"cycle_index": 1, "objective_state": "failed", "historical": True},
            {"cycle_index": 2, "objective_state": "in_progress", "historical": True},
        ),
        "domain": "odoo",
        "current_fields": {"path": "models/partner.py"},
    },
    # 7 repair → validation evidence → COMPLETE
    {
        "family": "cont_repair_validate_complete",
        "objective": "Ship models/partner.py field change for res.partner",
        "cycle": 3,
        "objective_state": "complete",
        "execution_state": "succeeded",
        "validation_status": "passed",
        "repair_status": "applied",
        "missing": (),
        "blockers": (),
        "hint": "complete",
        "loop": "complete",
        "reason": "Odoo objective complete after independent validation evidence on models/partner.py",
        "history": (
            {"cycle_index": 1, "objective_state": "incomplete", "historical": True},
            {"cycle_index": 2, "objective_state": "in_progress", "historical": True},
        ),
        "domain": "odoo",
        "current_fields": {"path": "models/partner.py"},
    },
    # 8 previous COMPLETE → new-cycle failure
    {
        "family": "cont_prev_complete_current_failure",
        "objective": "Ship hotfix after prior complete session",
        "cycle": 5,
        "objective_state": "failed",
        "execution_state": "failed",
        "validation_status": "failed",
        "missing": ("validation",),
        "failures": ("validation_failed",),
        "blockers": (),
        "hint": "replan",
        "loop": "replan",
        "reason": "Current-cycle failure is authoritative; prior COMPLETE must not leak",
        "history": (
            {
                "cycle_index": 4,
                "objective_state": "complete",
                "historical": True,
                "note": "prior_complete",
            },
        ),
        "domain": None,
        "current_fields": {"historical_summary_note": "prior_complete_must_not_overwrite"},
    },
    # 9 previous failure → current success
    {
        "family": "cont_prev_failure_current_success",
        "objective": "Recover Odoo module __manifest__.py after prior failure",
        "cycle": 2,
        "objective_state": "complete",
        "execution_state": "succeeded",
        "validation_status": "passed",
        "missing": (),
        "blockers": (),
        "hint": "complete",
        "loop": "complete",
        "reason": "Current Odoo success is authoritative; prior failure must not overwrite",
        "history": (
            {
                "cycle_index": 1,
                "objective_state": "failed",
                "historical": True,
                "note": "prior_failure",
            },
        ),
        "domain": "odoo",
        "current_fields": {
            "path": "addons/partner_ext/__manifest__.py",
            "historical_summary_note": "prior_failure_must_not_overwrite_current_success",
        },
    },
    # 10 empty current evidence
    {
        "family": "cont_empty_evidence",
        "objective": "Unknown early engineering objective",
        "cycle": 0,
        "objective_state": "unknown",
        "execution_state": "missing_observation",
        "observation_quality": "missing",
        "validation_status": "",
        "missing": ("observation", "validation"),
        "blockers": (),
        "hint": "clarify",
        "loop": "clarify",
        "reason": "Empty current evidence — cannot COMPLETE; need clarification",
        "history": (),
        "domain": None,
        "current_fields": {},
    },
    # 11 partial evidence
    {
        "family": "cont_partial_evidence",
        "objective": "Collect diagnostics for failing objective",
        "cycle": 1,
        "objective_state": "incomplete",
        "execution_state": "partial",
        "observation_quality": "partial",
        "validation_status": "not_run",
        "missing": ("full_diagnostics",),
        "blockers": (),
        "hint": "continue",
        "loop": "continue",
        "reason": "Partial diagnostics only — continue gathering evidence",
        "history": (),
        "domain": None,
    },
    # 12 conflicting evidence
    {
        "family": "cont_conflicting_evidence",
        "objective": "Decide after mixed observation quality",
        "cycle": 2,
        "objective_state": "incomplete",
        "execution_state": "partial",
        "observation_quality": "partial",
        "validation_status": "failed",
        "repair_status": "applied",
        "missing": ("consistent_validation",),
        "failures": ("validation_failed",),
        "warnings": ("repair_reported_ok_but_validation_failed",),
        "blockers": (),
        "hint": "replan",
        "loop": "replan",
        "reason": "Conflicting evidence — do not auto-complete or auto-repair",
        "history": ({"cycle_index": 1, "objective_state": "in_progress", "historical": True},),
        "domain": None,
    },
    # 13 blocker after previous progress
    {
        "family": "cont_blocker_after_progress",
        "objective": "Merge feature after progress",
        "cycle": 2,
        "objective_state": "blocked",
        "execution_state": "succeeded",
        "validation_status": "passed",
        "missing": (),
        "blockers": ("awaiting_approval",),
        "hint": "approve",
        "loop": "approve",
        "reason": "Progress made but blocker awaiting approval — wait/approve",
        "history": ({"cycle_index": 1, "objective_state": "in_progress", "historical": True},),
        "domain": None,
    },
    # 14 blocker resolved but objective still incomplete
    {
        "family": "cont_blocker_resolved_still_incomplete",
        "objective": "Import approved Odoo module artifact and finish objective",
        "cycle": 2,
        "objective_state": "incomplete",
        "execution_state": "succeeded",
        "validation_status": "pending",
        "missing": ("validation", "publish"),
        "blockers": (),
        "hint": "continue",
        "loop": "continue",
        "reason": "Approval blocker cleared for Odoo module; objective still incomplete",
        "history": (
            {
                "cycle_index": 1,
                "objective_state": "blocked",
                "historical": True,
                "note": "awaiting_approval",
            },
        ),
        "domain": "odoo",
        "current_fields": {"artifact": "addons/partner_ext", "module_manifest": "__manifest__.py"},
    },
    # 15 multiple Work Units → one objective
    {
        "family": "cont_multi_workunit_one_objective",
        "objective": "Inspect then compare repository before edit",
        "cycle": 2,
        "objective_state": "in_progress",
        "execution_state": "succeeded",
        "validation_status": "not_run",
        "missing": ("edit", "validation"),
        "blockers": (),
        "hint": "replan",
        "loop": "replan",
        "reason": "Multiple work units observed; objective not yet complete — plan next WHAT",
        "history": (
            {
                "cycle_index": 1,
                "objective_state": "in_progress",
                "historical": True,
                "note": "inspect_done",
            },
        ),
        "domain": None,
        "current_fields": {"completed_capabilities": ["repository.inspect", "repository.compare"]},
    },
    # 16 multiple observations before continuation
    {
        "family": "cont_multi_observation_continue",
        "objective": "Analyze problems after logs and diagnostics",
        "cycle": 2,
        "objective_state": "in_progress",
        "execution_state": "succeeded",
        "observation_quality": "succeeded",
        "validation_status": "not_run",
        "missing": ("analysis",),
        "blockers": (),
        "hint": "continue",
        "loop": "continue",
        "reason": "Multiple observations collected; continue toward analysis WHAT",
        "history": ({"cycle_index": 1, "objective_state": "incomplete", "historical": True},),
        "domain": None,
        "current_fields": {
            "observations": ["diagnostics.collect_logs", "diagnostics.collect_diagnostics"]
        },
    },
    # 17 evidence that does not justify repair
    {
        "family": "cont_evidence_no_repair",
        "objective": "Respond to skipped HTTP capability",
        "cycle": 1,
        "objective_state": "incomplete",
        "execution_state": "succeeded",
        "observation_quality": "succeeded",
        "validation_status": "not_run",
        "missing": ("alternate_path",),
        "blockers": (),
        "hint": "replan",
        "loop": "replan",
        "reason": "Capability skipped by policy — evidence does not justify repair",
        "history": (),
        "domain": None,
        "current_fields": {"last_status": "skipped", "capability_id": "communication.http"},
    },
    # 18 evidence justifying a new WHAT capability
    {
        "family": "cont_evidence_new_what",
        "objective": "Locate sale_order action_confirm methods then read models/sale.py",
        "cycle": 1,
        "objective_state": "incomplete",
        "execution_state": "succeeded",
        "observation_quality": "succeeded",
        "validation_status": "not_run",
        "missing": ("read",),
        "blockers": (),
        "hint": "replan",
        "loop": "replan",
        "reason": "Search evidence for action_confirm justifies next WHAT workspace.read — not repair",
        "history": (),
        "domain": "odoo",
        "current_fields": {
            "last_capability": "workspace.search",
            "match_count": 3,
            "path": "models/sale.py",
            "symbol": "action_confirm",
        },
    },
    # 19 continuation with bounded historical summary
    {
        "family": "cont_bounded_history_continue",
        "objective": "Continue long-running engineering objective",
        "cycle": 4,
        "objective_state": "in_progress",
        "execution_state": "succeeded",
        "validation_status": "pending",
        "missing": ("validation",),
        "blockers": (),
        "hint": "continue",
        "loop": "continue",
        "reason": "Bounded history present; current evidence still incomplete",
        "history": (
            {"cycle_index": 1, "objective_state": "incomplete", "historical": True},
            {"cycle_index": 2, "objective_state": "in_progress", "historical": True},
            {"cycle_index": 3, "objective_state": "in_progress", "historical": True},
        ),
        "domain": None,
    },
    # 20 history present but current evidence authoritative
    {
        "family": "cont_history_present_current_authoritative",
        "objective": "Override historical optimism with current failure",
        "cycle": 3,
        "objective_state": "failed",
        "execution_state": "failed",
        "validation_status": "failed",
        "missing": (),
        "failures": ("validation_failed",),
        "blockers": (),
        "hint": "replan",
        "loop": "replan",
        "reason": "History showed progress; current evidence is failed — current wins",
        "history": (
            {"cycle_index": 1, "objective_state": "in_progress", "historical": True},
            {"cycle_index": 2, "objective_state": "in_progress", "historical": True},
        ),
        "domain": None,
        "current_fields": {"authoritative": "current_evidence"},
    },
    # Additional diversity (not trivial clones)
    {
        "family": "cont_retry_transient",
        "objective": "Retry timed-out program execution",
        "cycle": 2,
        "objective_state": "incomplete",
        "execution_state": "timed_out",
        "validation_status": "not_run",
        "missing": ("program_result",),
        "failures": ("timed_out",),
        "blockers": (),
        "hint": "retry",
        "loop": "retry",
        "reason": "Transient timeout with unchanged objective — retry",
        "history": ({"cycle_index": 1, "objective_state": "incomplete", "historical": True},),
        "domain": None,
    },
    {
        "family": "cont_recover_session",
        "objective": "Recover engineering session after interruption",
        "cycle": 2,
        "objective_state": "in_progress",
        "execution_state": "partial",
        "validation_status": "not_run",
        "missing": ("observation",),
        "blockers": (),
        "hint": "recover",
        "loop": "recover",
        "reason": "Session recoverable; re-observe before deciding WHAT",
        "history": ({"cycle_index": 1, "objective_state": "incomplete", "historical": True},),
        "domain": None,
    },
    {
        "family": "cont_cancel_operator",
        "objective": "Honour operator cancellation",
        "cycle": 1,
        "objective_state": "cancelled",
        "execution_state": "cancelled",
        "validation_status": "",
        "missing": (),
        "blockers": (),
        "hint": "cancel",
        "loop": "cancel",
        "reason": "Operator cancelled the engineering session",
        "history": (),
        "domain": None,
    },
    {
        "family": "cont_pause_dependency_window",
        "objective": "Pause until external dependency window opens",
        "cycle": 2,
        "objective_state": "blocked",
        "execution_state": "succeeded",
        "validation_status": "passed",
        "missing": (),
        "blockers": ("dependency_window_closed",),
        "hint": "pause",
        "loop": "pause",
        "reason": "Blocked on external window — pause rather than fabricate repair",
        "history": ({"cycle_index": 1, "objective_state": "in_progress", "historical": True},),
        "domain": None,
    },
    {
        "family": "cont_reject_approval",
        "objective": "Handle rejected change set",
        "cycle": 2,
        "objective_state": "incomplete",
        "execution_state": "succeeded",
        "validation_status": "passed",
        "missing": (),
        "blockers": (),
        "failures": ("approval_rejected",),
        "hint": "replan",
        "loop": "reject",
        "reason": "Reviewer rejected proposed change set — return to planning",
        "history": ({"cycle_index": 1, "objective_state": "blocked", "historical": True},),
        "domain": None,
    },
    {
        "family": "cont_modify_approval",
        "objective": "Incorporate reviewer modification requests for models/partner.py",
        "cycle": 2,
        "objective_state": "incomplete",
        "execution_state": "succeeded",
        "validation_status": "passed",
        "missing": ("modifications",),
        "blockers": (),
        "hint": "replan",
        "loop": "modify",
        "reason": "Approval requests modification of Odoo models/partner.py before accept",
        "history": ({"cycle_index": 1, "objective_state": "blocked", "historical": True},),
        "domain": "odoo",
        "current_fields": {"path": "models/partner.py"},
    },
)


def generate_continuity_expansion() -> list[dict[str, Any]]:
    """Emit state + decision_context + loop triples for each continuity scenario.

    Expands each base scenario into a few related cycle variants without
    fabricating automatic repair pipelines.
    """
    out: list[dict[str, Any]] = []
    seq = 0
    # Repeat families with distinct cycle/objective variants for volume
    variants: list[dict[str, Any]] = []
    for spec in _SCENARIOS:
        variants.append(spec)
        # Sibling: same decision shape, distinct objective surface (not trivial string swap)
        sib = {
            **spec,
            "family": f"{spec['family']}_alt_surface",
            "objective": f"{spec['objective']} [surface: diagnostics-led]",
            "cycle": int(spec["cycle"]),
            "current_fields": {
                **dict(spec.get("current_fields") or {}),
                "surface": "diagnostics_led",
            },
        }
        variants.append(sib)
        if spec["family"] in {
            "cont_op_ok_obj_incomplete_replan",
            "cont_fail_replan",
            "cont_fail_escalate",
            "cont_empty_evidence",
            "cont_prev_complete_current_failure",
            "cont_prev_failure_current_success",
            "cont_conflicting_evidence",
            "cont_evidence_no_repair",
            "cont_repair_validate_still_incomplete",
            "cont_repair_validate_complete",
        }:
            v3 = {
                **spec,
                "family": f"{spec['family']}_alt_repo",
                "objective": f"{spec['objective']} [surface: repository-led]",
                "current_fields": {
                    **dict(spec.get("current_fields") or {}),
                    "surface": "repository_led",
                },
            }
            variants.append(v3)

    for spec in variants:
        seq += 1
        domain = spec.get("domain") if isinstance(spec.get("domain"), str) else None
        family = str(spec["family"])
        meta = {
            "scenario_family": family,
            "scenario": family,
            "controlled_batch": TR7_BATCH_VERSION,
            "tr7_continuity": True,
        }
        provider = "planner"
        if spec.get("loop") in {"approve", "reject", "modify"}:
            provider = "approval"
        elif spec.get("loop") == "clarify":
            provider = "conversation"

        st = EngineeringStateRecord(
            record_type="engineering_state",
            record_id=_rid("st", seq),
            objective=str(spec["objective"]),
            objective_state=str(spec["objective_state"]),
            session_state="active"
            if spec["objective_state"] not in {"cancelled", "escalated"}
            else str(spec["objective_state"]),
            completion_state="ready" if spec["objective_state"] == "complete" else "open",
            cycle_index=int(spec["cycle"]),
            current_fields=dict(spec.get("current_fields") or {}),
            provider_capability="planner",
            domain_specialization=domain,
            metadata=fixture_metadata(
                generator="tr7_continuity",
                index=seq,
                provider_capability="planner",
                domain_specialization=domain,
                extra=meta,
            ),
        )
        # Enrich current_fields with validation/repair for continuity tests
        cf = dict(st.current_fields)
        if spec.get("validation_status") is not None:
            cf.setdefault("validation_status", spec.get("validation_status"))
        if spec.get("repair_status") is not None:
            cf.setdefault("repair_status", spec.get("repair_status"))
        st = EngineeringStateRecord(
            record_type="engineering_state",
            record_id=st.record_id,
            objective=st.objective,
            objective_state=st.objective_state,
            session_state=st.session_state,
            completion_state=st.completion_state,
            cycle_index=st.cycle_index,
            current_fields=cf,
            provider_capability=st.provider_capability,
            domain_specialization=st.domain_specialization,
            metadata=st.metadata,
        )
        out.append(st.to_dict())

        dc = DecisionContextRecord(
            record_type="decision_context",
            record_id=_rid("dc", seq),
            objective=str(spec["objective"]),
            objective_state=str(spec["objective_state"]),
            cycle_index=int(spec["cycle"]),
            execution_state=str(spec.get("execution_state") or ""),
            observation_quality=str(spec.get("observation_quality") or ""),
            validation_status=str(spec.get("validation_status") or ""),
            repair_status=str(spec.get("repair_status") or ""),
            missing_outcomes=tuple(spec.get("missing") or ()),
            blockers=tuple(spec.get("blockers") or ()),
            failures=tuple(spec.get("failures") or ()),
            possible_next_actions=(str(spec["hint"]), "escalate", "cancel"),
            continuation_hint=str(spec["hint"]),
            bounded_history=tuple(spec.get("history") or ()),
            provider_capability="planner" if provider != "conversation" else "conversation",
            domain_specialization=domain,
            metadata=fixture_metadata(
                generator="tr7_continuity",
                index=seq,
                provider_capability="planner" if provider != "conversation" else "conversation",
                domain_specialization=domain,
                extra=meta,
            ),
        )
        out.append(dc.to_dict())

        ld = LoopDecisionRecord(
            record_type="loop_decision",
            record_id=_rid("ld", seq),
            decision_kind=str(spec["loop"]),
            reason=str(spec["reason"]),
            next_goal=""
            if spec["loop"] in {"complete", "escalate", "cancel", "pause"}
            else "Continue from current evidence",
            provider_capability=provider if provider in {"planner", "approval", "conversation"} else "planner",
            domain_specialization=domain,
            metadata=fixture_metadata(
                generator="tr7_continuity",
                index=seq,
                provider_capability=provider if provider in {"planner", "approval", "conversation"} else "planner",
                domain_specialization=domain,
                extra=meta,
            ),
        )
        out.append(ld.to_dict())

    return out
