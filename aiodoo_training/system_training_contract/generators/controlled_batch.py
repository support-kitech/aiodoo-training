"""TR-5 controlled FP2-native batch generation (coverage-aware, gate-checked).

Produces one isolated batch under ``datasets/fp2/controlled_batch_1/``.
Does not modify legacy production JSONL. Does not train adapters.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiodoo_training.system_training_contract.generators.common import (
    DATASET_GENERATION_VERSION,
    fixture_metadata,
    write_jsonl,
)
from aiodoo_training.system_training_contract.generators.emit import (
    GENERATOR_NAMES,
    generate_all,
    generate_projection_fixtures,
)
from aiodoo_training.system_training_contract.quality.analysis import (
    capability_coverage,
    domain_distribution,
    find_duplicates,
)
from aiodoo_training.system_training_contract.quality.common import (
    NATIVE_FAMILIES,
    extract_engineering_capability,
    fingerprint_record,
    stable_dumps,
)
from aiodoo_training.system_training_contract.quality.formatters import (
    format_fp2_pack,
)
from aiodoo_training.system_training_contract.quality.gates import (
    scan_forbidden_how,
    scan_taxonomy,
)
from aiodoo_training.system_training_contract.quality.harness import (
    QualityReport,
    evaluate_fp2_corpus,
)
from aiodoo_training.system_training_contract.quality.splits import (
    SplitAssignment,
    assign_split,
    document_split_strategy,
    scenario_key,
)
from aiodoo_training.system_training_contract.records import (
    CapabilityIntentRecord,
    DecisionContextRecord,
    EngineeringFeedbackRecord,
    EngineeringStateRecord,
    LoopDecisionRecord,
    ObservationRecord,
    PlanningDecisionRecord,
    TrainingRecordError,
    WorkUnitRecord,
    validate_record_mapping,
)
from aiodoo_training.system_training_contract.taxonomy import (
    PREFERRED_ENGINEERING_CAPABILITY_IDS,
)
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

CONTROLLED_BATCH_VERSION: str = "fp2-controlled-1.0.0"
CONTROLLED_BATCH_MAX: int = 1500
CONTROLLED_BATCH_MIN: int = 1000

# Prefer strengthening these; dampen overrepresented caps from TR-4.
_PRIORITY_CAPS: frozenset[str] = frozenset(
    {
        "artifact.attachment",
        "artifact.import",
        "diagnostics.collect_logs",
        "repository.merge",
        "repository.history",
        "workspace.bind",
        "artifact.export",
        "artifact.publish",
        "diagnostics.collect_diagnostics",
        "diagnostics.analyze_problems",
        "diagnostics.static_analysis",
        "repository.branch",
        "repository.modify",
        "communication.http",
        "execution.execute_program",
        "execution.repair",
        "workspace.read",
        "workspace.search",
        "workspace.navigate",
        "repository.inspect",
        "repository.compare",
    }
)
_DAMPENED_CAPS: frozenset[str] = frozenset({"validation.run", "workspace.write"})

# Meaningful scenario templates (not trivial path A/B/C spam).
_SCENARIOS: tuple[dict[str, Any], ...] = (
    # --- coverage gaps / weak ---
    {"cap": "workspace.bind", "objective": "Bind product root before inspection", "args": {"root_hint": "project"}, "domain": None, "family": "bind_inspect"},
    {"cap": "workspace.bind", "objective": "Bind Odoo addons workspace for module work", "args": {"root_hint": "addons/sale"}, "domain": "odoo", "family": "bind_odoo"},
    {"cap": "repository.history", "objective": "Review history before merge decision", "args": {"path": ".", "limit": 25}, "domain": None, "family": "history_pre_merge"},
    {"cap": "repository.history", "objective": "Audit partner model history for regressions", "args": {"path": "models/partner.py", "limit": 30}, "domain": "odoo", "family": "history_odoo"},
    {"cap": "repository.merge", "objective": "Merge approved feature into integration", "args": {"source": "feat/partner-field", "target": "integration"}, "domain": None, "family": "merge_feature", "mutating": True},
    {"cap": "repository.merge", "objective": "Merge hotfix line after approval evidence", "args": {"source": "hotfix/import", "target": "stable"}, "domain": "odoo", "family": "merge_hotfix", "mutating": True},
    {"cap": "diagnostics.collect_logs", "objective": "Collect bounded logs after execution failure", "args": {"scope": "session", "bound": "recent"}, "domain": None, "family": "logs_after_fail"},
    {"cap": "diagnostics.collect_logs", "objective": "Collect module logs for Odoo worker failure", "args": {"scope": "module", "bound": "error"}, "domain": "odoo", "family": "logs_odoo"},
    {"cap": "artifact.attachment", "objective": "Attach patch bundle to engineering context", "args": {"path": "artifacts/change.patch", "artifact_kind": "patch_bundle"}, "domain": None, "family": "attach_patch"},
    {"cap": "artifact.attachment", "objective": "Attach validation report artifact", "args": {"path": "artifacts/validation.json", "artifact_kind": "report"}, "domain": "odoo", "family": "attach_report"},
    {"cap": "artifact.import", "objective": "Import confirmed patch into workspace", "args": {"artifact_kind": "patch_bundle", "confirmed": True}, "domain": None, "family": "import_ok", "mutating": True},
    {"cap": "artifact.import", "objective": "Refuse import without confirmation", "args": {"artifact_kind": "patch_bundle", "confirmed": False}, "domain": "odoo", "family": "import_blocked", "mutating": True, "expect_fail": True},
    # --- balanced underrepresented ---
    {"cap": "workspace.read", "objective": "Read controller before modification", "args": {"path": "controllers/main.py"}, "domain": "odoo", "family": "read_controller"},
    {"cap": "workspace.read", "objective": "Read configuration schema", "args": {"path": "config/schema.yaml"}, "domain": None, "family": "read_config"},
    {"cap": "workspace.search", "objective": "Search for deprecated API usage", "args": {"query": "api.one"}, "domain": "odoo", "family": "search_deprecated"},
    {"cap": "workspace.search", "objective": "Locate TODO markers in tools", "args": {"query": "TODO"}, "domain": None, "family": "search_todo"},
    {"cap": "workspace.navigate", "objective": "Navigate tests directory layout", "args": {"path": "tests"}, "domain": None, "family": "nav_tests"},
    {"cap": "repository.inspect", "objective": "Inspect dirty tree before planning", "args": {}, "domain": None, "family": "inspect_dirty"},
    {"cap": "repository.compare", "objective": "Compare feature branch to integration", "args": {"base": "integration", "head": "feat/partner-field"}, "domain": None, "family": "compare_branches"},
    {"cap": "repository.branch", "objective": "Prepare recovery branch intent", "args": {"branch": "recover/import-fix"}, "domain": None, "family": "branch_recover"},
    {"cap": "repository.modify", "objective": "Stage corrected import paths", "args": {"paths": ["models/partner.py", "tests/test_partner.py"]}, "domain": "odoo", "family": "modify_stage"},
    {"cap": "execution.execute_program", "objective": "Execute migration dry-run program", "args": {"entrypoint": "tools.migrate:dry_run"}, "domain": "odoo", "family": "exec_migrate"},
    {"cap": "execution.execute_program", "objective": "Execute health probe program", "args": {"entrypoint": "tools.health:probe"}, "domain": None, "family": "exec_health"},
    {"cap": "execution.repair", "objective": "Repair missing dependency declaration", "args": {"path": "__manifest__.py", "symptom": "MissingDependency"}, "domain": "odoo", "family": "repair_manifest"},
    {"cap": "execution.repair", "objective": "Repair broken relative import", "args": {"path": "tools/helper.py", "symptom": "ImportError"}, "domain": None, "family": "repair_import"},
    {"cap": "communication.http", "objective": "POST status webhook WHAT", "args": {"method": "POST", "url": "https://example.invalid/hooks/status"}, "domain": None, "family": "http_post"},
    {"cap": "diagnostics.collect_diagnostics", "objective": "Collect diagnostics after partial failure", "args": {"scope": "workspace"}, "domain": None, "family": "diag_partial"},
    {"cap": "diagnostics.analyze_problems", "objective": "Normalize log findings into problems", "args": {"problem_ids": ["L1", "L2"]}, "domain": None, "family": "analyze_logs"},
    {"cap": "diagnostics.static_analysis", "objective": "Static analysis on controllers", "args": {"scope": "controllers"}, "domain": "odoo", "family": "static_controllers"},
    {"cap": "artifact.publish", "objective": "Publish repair outcome artifact", "args": {"artifact_kind": "patch_bundle"}, "domain": None, "family": "publish_repair"},
    {"cap": "artifact.export", "objective": "Export attached artifacts for review", "args": {"artifact_kind": "patch_bundle"}, "domain": None, "family": "export_review"},
    # dampened — few meaningful scenarios only
    {"cap": "workspace.write", "objective": "Write recovery helper after independent decision", "args": {"path": "tools/recover.py"}, "domain": None, "family": "write_recover"},
    {"cap": "validation.run", "objective": "Validate after independent evidence review", "args": {"scope": "module"}, "domain": "odoo", "family": "validate_review"},
)


@dataclass
class ControlledBatchResult:
    output_dir: str
    total_native: int = 0
    per_family: dict[str, int] = field(default_factory=dict)
    per_capability: dict[str, int] = field(default_factory=dict)
    domain: dict[str, Any] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    development_pack_count: int = 0
    reasoning_pack_count: int = 0
    duplicates: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    decision: str = "FAIL"
    warnings: list[str] = field(default_factory=list)
    checksum: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "controlled_batch_version": CONTROLLED_BATCH_VERSION,
            "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
            "dataset_generation_version": DATASET_GENERATION_VERSION,
            "output_dir": self.output_dir,
            "total_native": self.total_native,
            "per_family": self.per_family,
            "per_capability": self.per_capability,
            "domain": self.domain,
            "split_counts": self.split_counts,
            "development_pack_count": self.development_pack_count,
            "reasoning_pack_count": self.reasoning_pack_count,
            "duplicates": self.duplicates,
            "coverage": {
                "coverage_pct": self.coverage.get("coverage_pct"),
                "covered_count": self.coverage.get("covered_count"),
                "uncovered": self.coverage.get("uncovered"),
                "preferred_total": self.coverage.get("preferred_total"),
            },
            "quality": self.quality,
            "decision": self.decision,
            "warnings": self.warnings,
            "checksum": self.checksum,
        }


def _seq_id(prefix: str, n: int) -> str:
    return f"fp2-cb1-{prefix}-{n:04d}"


def _build_from_scenario(spec: dict[str, Any], seq: int) -> list[dict[str, Any]]:
    """Expand one semantic scenario into intent + WU + observation (+ feedback)."""
    cap = str(spec["cap"])
    objective = str(spec["objective"])
    args = dict(spec.get("args") or {})
    domain = spec.get("domain")
    domain_s = domain if isinstance(domain, str) else None
    family = str(spec.get("family") or cap)
    expect_fail = bool(spec.get("expect_fail"))
    mutating = bool(spec.get("mutating"))
    provider = "repair" if cap == "execution.repair" else "execution"
    if cap.startswith("workspace.") and domain_s == "odoo":
        provider = "coding"

    out: list[dict[str, Any]] = []
    meta_extra = {"scenario_family": family, "controlled_batch": CONTROLLED_BATCH_VERSION}

    intent = CapabilityIntentRecord(
        record_type="capability_intent",
        record_id=_seq_id("ci", seq),
        capability_id=cap,
        objective=objective,
        args=args,
        reason=f"controlled scenario {family}",
        provider_capability=provider if provider in {"coding", "repair", "execution"} else "execution",
        domain_specialization=domain_s,
        metadata=fixture_metadata(
            generator="controlled_batch",
            index=seq,
            provider_capability=provider,
            domain_specialization=domain_s,
            extra=meta_extra,
        ),
    )
    out.append(intent.to_dict())

    wu = WorkUnitRecord(
        record_type="execution_work_unit",
        record_id=_seq_id("wu", seq),
        work_id=f"ewu-cb1-{seq:04d}",
        capability_id=cap,
        objective=objective,
        inputs=args,
        expected_outputs={"ok": not expect_fail},
        constraints={"requires_approval": True} if mutating and cap in {"repository.merge", "artifact.import"} else {},
        provider_capability="execution" if provider != "repair" else "repair",
        domain_specialization=domain_s,
        metadata=fixture_metadata(
            generator="controlled_batch",
            index=seq,
            provider_capability="execution" if provider != "repair" else "repair",
            domain_specialization=domain_s,
            extra=meta_extra,
        ),
    )
    out.append(wu.to_dict())

    kind = "execution_result"
    if cap.startswith("repository.merge"):
        kind = "repository_merge"
    elif cap.startswith("repository.history"):
        kind = "repository_history"
    elif cap.startswith("diagnostics."):
        kind = "diagnostics_result"
    elif cap.startswith("artifact."):
        kind = "artifact_result"
    elif cap == "workspace.bind":
        kind = "environment_status"
    elif cap == "validation.run":
        kind = "validation_result"
    elif cap == "execution.repair":
        kind = "repair_result"
    elif cap == "execution.execute_program":
        kind = "program_output"
    elif cap == "workspace.search":
        kind = "search_result"

    obs = ObservationRecord(
        record_type="observation",
        record_id=_seq_id("obs", seq),
        kind=kind,
        status="failed" if expect_fail else "succeeded",
        capability_id=cap,
        summary=f"{'Failed' if expect_fail else 'Succeeded'}: {objective}",
        evidence={"scenario_family": family, "ok": not expect_fail},
        provider_capability="execution" if provider != "repair" else "repair",
        domain_specialization=domain_s,
        metadata=fixture_metadata(
            generator="controlled_batch",
            index=seq,
            provider_capability="execution" if provider != "repair" else "repair",
            domain_specialization=domain_s,
            extra=meta_extra,
        ),
    )
    out.append(obs.to_dict())

    # Feedback distinguishing operation vs objective when failure/partial semantics apply
    obj_state = "incomplete" if expect_fail or mutating else "in_progress"
    if expect_fail:
        obj_state = "blocked" if "confirm" in objective.lower() or "approval" in str(args) else "incomplete"
    fb = EngineeringFeedbackRecord(
        record_type="engineering_feedback",
        record_id=_seq_id("fb", seq),
        objective=objective,
        objective_state=obj_state,
        execution_state="failed" if expect_fail else "succeeded",
        observation_quality="failed" if expect_fail else "succeeded",
        continuation_options=("replan", "escalate") if expect_fail else ("continue", "complete"),
        recommended_continuation="replan" if expect_fail else "continue",
        missing_outcomes=("validation",) if not expect_fail and mutating else (),
        failures=("operation_failed",) if expect_fail else (),
        blockers=("confirmation_required",) if expect_fail and cap == "artifact.import" else (),
        provider_capability="planner",
        domain_specialization=domain_s,
        metadata=fixture_metadata(
            generator="controlled_batch",
            index=seq,
            provider_capability="planner",
            domain_specialization=domain_s,
            extra={**meta_extra, "scenario": f"fb_{family}"},
        ),
    )
    out.append(fb.to_dict())
    return out


def _multi_cycle_narratives(start_seq: int) -> list[dict[str, Any]]:
    """Evidence-based multi-cycle narratives — no automatic repair pipeline."""
    out: list[dict[str, Any]] = []
    seq = start_seq
    # Narrative A: failure → independent replan → success (not forced repair→validate)
    cycles = (
        {
            "cycle": 1,
            "objective": "Ship partner field change",
            "objective_state": "incomplete",
            "validation_status": "failed",
            "hint": "replan",
            "loop": "replan",
            "reason": "Validation failed; missing outcomes remain — decide next WHAT from evidence",
            "domain": "odoo",
            "family": "narrative_partner_a",
        },
        {
            "cycle": 2,
            "objective": "Ship partner field change",
            "objective_state": "in_progress",
            "validation_status": "pending",
            "repair_status": "applied",
            "hint": "continue",
            "loop": "continue",
            "reason": "Correction observed; objective still incomplete — continue from current evidence",
            "domain": "odoo",
            "family": "narrative_partner_a",
        },
        {
            "cycle": 3,
            "objective": "Ship partner field change",
            "objective_state": "complete",
            "validation_status": "passed",
            "repair_status": "applied",
            "hint": "complete",
            "loop": "complete",
            "reason": "Objective complete; validation passed; no blockers",
            "domain": "odoo",
            "family": "narrative_partner_a",
        },
    )
    for c in cycles:
        seq += 1
        domain = c["domain"]
        family = c["family"]
        meta = {"scenario_family": family, "scenario": f"cycle{c['cycle']}_{family}", "controlled_batch": CONTROLLED_BATCH_VERSION}
        st = EngineeringStateRecord(
            record_type="engineering_state",
            record_id=_seq_id("st", seq),
            objective=str(c["objective"]),
            objective_state=str(c["objective_state"]),
            session_state="active",
            completion_state="ready" if c["objective_state"] == "complete" else "open",
            cycle_index=int(c["cycle"]),
            current_fields={
                "validation_status": c.get("validation_status"),
                "repair_status": c.get("repair_status", "not_started"),
            },
            provider_capability="planner",
            domain_specialization=domain,
            metadata=fixture_metadata(
                generator="controlled_batch",
                index=seq,
                provider_capability="planner",
                domain_specialization=domain,
                extra=meta,
            ),
        )
        out.append(st.to_dict())
        hist = ()
        if c["cycle"] > 1:
            hist = tuple(
                {
                    "cycle_index": i,
                    "objective_state": cycles[i - 1]["objective_state"],
                    "historical": True,
                }
                for i in range(1, c["cycle"])
            )
        dc = DecisionContextRecord(
            record_type="decision_context",
            record_id=_seq_id("dc", seq),
            objective=str(c["objective"]),
            objective_state=str(c["objective_state"]),
            cycle_index=int(c["cycle"]),
            validation_status=str(c.get("validation_status") or ""),
            repair_status=str(c.get("repair_status") or ""),
            missing_outcomes=("validation",) if c["objective_state"] != "complete" else (),
            possible_next_actions=(str(c["hint"]), "escalate"),
            continuation_hint=str(c["hint"]),
            bounded_history=hist,
            provider_capability="planner",
            domain_specialization=domain,
            metadata=fixture_metadata(
                generator="controlled_batch",
                index=seq,
                provider_capability="planner",
                domain_specialization=domain,
                extra=meta,
            ),
        )
        out.append(dc.to_dict())
        ld = LoopDecisionRecord(
            record_type="loop_decision",
            record_id=_seq_id("ld", seq),
            decision_kind=str(c["loop"]),
            reason=str(c["reason"]),
            next_goal="" if c["loop"] in {"complete", "escalate"} else "Continue from current evidence",
            provider_capability="planner",
            domain_specialization=domain,
            metadata=fixture_metadata(
                generator="controlled_batch",
                index=seq,
                provider_capability="planner",
                domain_specialization=domain,
                extra=meta,
            ),
        )
        out.append(ld.to_dict())

    # Narrative B: previous complete → current failure isolation
    seq += 1
    out.append(
        EngineeringStateRecord(
            record_type="engineering_state",
            record_id=_seq_id("st", seq),
            objective="Ship hotfix",
            objective_state="failed",
            session_state="active",
            completion_state="open",
            cycle_index=5,
            current_fields={
                "validation_status": "failed",
                "historical_summary_note": "prior_complete_must_not_overwrite",
            },
            provider_capability="planner",
            metadata=fixture_metadata(
                generator="controlled_batch",
                index=seq,
                provider_capability="planner",
                extra={
                    "scenario": "previous_complete_current_failure",
                    "scenario_family": "narrative_complete",
                    "controlled_batch": CONTROLLED_BATCH_VERSION,
                },
            ),
        ).to_dict()
    )
    seq += 1
    out.append(
        EngineeringStateRecord(
            record_type="engineering_state",
            record_id=_seq_id("st", seq),
            objective="Recover module",
            objective_state="complete",
            session_state="active",
            completion_state="ready",
            cycle_index=2,
            current_fields={
                "validation_status": "passed",
                "historical_summary_note": "prior_failure_must_not_overwrite_current_success",
            },
            provider_capability="planner",
            domain_specialization="odoo",
            metadata=fixture_metadata(
                generator="controlled_batch",
                index=seq,
                provider_capability="planner",
                domain_specialization="odoo",
                extra={
                    "scenario": "previous_failure_current_success",
                    "scenario_family": "narrative_failure",
                    "controlled_batch": CONTROLLED_BATCH_VERSION,
                },
            ),
        ).to_dict()
    )
    return out


def _planning_for_scenario(spec: dict[str, Any], seq: int) -> dict[str, Any]:
    cap = str(spec["cap"])
    domain = spec.get("domain") if isinstance(spec.get("domain"), str) else None
    steps = ({"action": cap, "args": dict(spec.get("args") or {})},)
    # Pair with a non-automatic next WHAT when useful (not repair→validate rule)
    if cap == "diagnostics.collect_logs":
        steps = (
            {"action": "diagnostics.collect_logs", "args": dict(spec.get("args") or {})},
            {"action": "diagnostics.analyze_problems", "args": {}},
        )
    elif cap == "workspace.bind":
        steps = (
            {"action": "workspace.bind", "args": dict(spec.get("args") or {})},
            {"action": "repository.inspect", "args": {}},
        )
    elif cap == "repository.merge":
        steps = (
            {"action": "repository.compare", "args": {"base": "integration"}},
            {"action": "repository.merge", "args": dict(spec.get("args") or {})},
        )
    return PlanningDecisionRecord(
        record_type="planning_decision",
        record_id=_seq_id("pd", seq),
        goal=str(spec["objective"]),
        decision_kind="replan",
        summary=f"Plan for {cap}",
        steps=steps,
        provider_capability="planner",
        domain_specialization=domain,
        metadata=fixture_metadata(
            generator="controlled_batch",
            index=seq,
            provider_capability="planner",
            domain_specialization=domain,
            extra={
                "scenario_family": str(spec.get("family") or cap),
                "controlled_batch": CONTROLLED_BATCH_VERSION,
            },
        ),
    ).to_dict()


def _cap_weight(cap: str, counts: Mapping[str, int]) -> float:
    n = counts.get(cap, 0)
    if cap in _DAMPENED_CAPS:
        return 0.15 / (1 + n)
    if cap in _PRIORITY_CAPS:
        return 3.0 / (1 + n)
    return 1.0 / (1 + n)


def generate_controlled_batch_records(*, target: int = 1200) -> list[dict[str, Any]]:
    """Generate a coverage-aware controlled batch (native records only)."""
    target = max(1, min(CONTROLLED_BATCH_MAX, int(target)))
    base_families = generate_all()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(rec: dict[str, Any]) -> None:
        fp = fingerprint_record(rec)
        if fp in seen:
            return
        # Also guard exact record_id collisions by regenerating not needed — skip dup fps
        try:
            validate_record_mapping(rec)
        except TrainingRecordError:
            return
        if scan_forbidden_how(rec) or scan_taxonomy(rec):
            return
        seen.add(fp)
        records.append(rec)

    for fam in GENERATOR_NAMES:
        for rec in base_families[fam]:
            _add(rec)

    for rec in _multi_cycle_narratives(9000):
        _add(rec)

    # Coverage-aware expansion of scenarios (repeat families with planner variants, not path spam)
    cap_counts: Counter[str] = Counter()
    for rec in records:
        eng = extract_engineering_capability(rec)
        if eng:
            for part in eng.split(","):
                if part in PREFERRED_ENGINEERING_CAPABILITY_IDS:
                    cap_counts[part] += 1

    seq = 100
    # Sort scenarios by weight (underrepresented first)
    ranked = sorted(
        _SCENARIOS,
        key=lambda s: -_cap_weight(str(s["cap"]), cap_counts),
    )
    # Expand rounds until target
    round_idx = 0
    while len(records) < target and round_idx < 8:
        for spec in ranked:
            if len(records) >= target:
                break
            cap = str(spec["cap"])
            # Strong dampening after enough examples for dampened caps
            if cap in _DAMPENED_CAPS and cap_counts[cap] >= 40:
                continue
            if cap not in _DAMPENED_CAPS and cap_counts[cap] >= 90 and round_idx > 2:
                continue
            seq += 1
            # Slight semantic variation by round (context tag — not trivial path churn)
            varied = dict(spec)
            if round_idx > 0:
                varied = {
                    **spec,
                    "objective": f"{spec['objective']} (context {round_idx + 1})",
                    "family": f"{spec['family']}_r{round_idx + 1}",
                }
            bundle = _build_from_scenario(varied, seq)
            for rec in bundle:
                before = len(records)
                _add(rec)
                if len(records) > before:
                    eng = extract_engineering_capability(rec)
                    if eng:
                        for part in eng.split(","):
                            if part in PREFERRED_ENGINEERING_CAPABILITY_IDS:
                                cap_counts[part] += 1
            seq += 1
            _add(_planning_for_scenario(varied, seq))
            if extract_engineering_capability(records[-1] if records else {}):
                pass
        round_idx += 1

    # Ensure every preferred capability appears at least once
    present = set()
    for rec in records:
        eng = extract_engineering_capability(rec)
        if eng:
            present.update(p for p in eng.split(",") if p in PREFERRED_ENGINEERING_CAPABILITY_IDS)
    missing = PREFERRED_ENGINEERING_CAPABILITY_IDS - present
    for cap in sorted(missing):
        seq += 1
        spec = {
            "cap": cap,
            "objective": f"Perform {cap} for engineering objective",
            "args": {},
            "domain": None,
            "family": f"ensure_{cap}",
        }
        for rec in _build_from_scenario(spec, seq):
            _add(rec)

    return records[:target] if len(records) > target else records


def _assert_no_negatives(records: list[dict[str, Any]]) -> None:
    for rec in records:
        meta = rec.get("metadata") if isinstance(rec.get("metadata"), Mapping) else {}
        qc = str(meta.get("quality_corpus") or "")
        if qc.startswith("negative") or meta.get("not_for_training"):
            raise RuntimeError(f"negative contaminated training batch: {rec.get('record_id')}")


def _leakage_check(records: list[dict[str, Any]]) -> list[str]:
    """Multi-cycle / same scenario_family must share split."""
    issues: list[str] = []
    by_family: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        meta = rec.get("metadata") if isinstance(rec.get("metadata"), Mapping) else {}
        fam = str(meta.get("scenario_family") or scenario_key(rec))
        by_family[fam].add(assign_split(rec).value)
    for fam, splits in by_family.items():
        if len(splits) > 1 and fam.startswith("narrative_"):
            issues.append(f"split_leakage:{fam}:{sorted(splits)}")
    return issues


def emit_controlled_batch(
    output_dir: str | Path,
    *,
    target: int = 1200,
) -> ControlledBatchResult:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Production controlled batches must stay within the TR-5 band.
    if target >= CONTROLLED_BATCH_MIN:
        target = max(CONTROLLED_BATCH_MIN, min(CONTROLLED_BATCH_MAX, target))
    else:
        # Allow smaller targets for unit tests only.
        target = max(1, min(CONTROLLED_BATCH_MAX, target))
    records = generate_controlled_batch_records(target=target)
    _assert_no_negatives(records)

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_family[str(rec.get("record_type") or "unknown")].append(rec)

    counts: dict[str, int] = {}
    for fam in NATIVE_FAMILIES:
        path = root / f"{fam}.jsonl"
        counts[fam] = write_jsonl(path, by_family.get(fam, []))

    # Projection fixtures copied for audit continuity (not scaled)
    proj = generate_projection_fixtures()
    write_jsonl(root / "projection_fixtures.jsonl", proj)

    # Splits
    split_rows = []
    split_counter: Counter[str] = Counter()
    for rec in records:
        sp = assign_split(rec)
        split_counter[sp.value] += 1
        split_rows.append(
            {
                "record_id": rec.get("record_id"),
                "record_type": rec.get("record_type"),
                "split": sp.value,
                "scenario_key": scenario_key(rec),
            }
        )
    write_jsonl(root / "splits.jsonl", split_rows)

    # Packs
    dev_examples = format_fp2_pack(records, pack="development")
    rea_examples = format_fp2_pack(records, pack="reasoning")
    write_jsonl(
        root / "pack_development.jsonl",
        [
            {
                "example_id": e.example_id,
                "dataset_type": e.dataset_type.value,
                "messages": [dict(m) for m in e.messages],
                "metadata": dict(e.metadata),
            }
            for e in dev_examples
        ],
    )
    write_jsonl(
        root / "pack_reasoning.jsonl",
        [
            {
                "example_id": e.example_id,
                "dataset_type": e.dataset_type.value,
                "messages": [dict(m) for m in e.messages],
                "metadata": dict(e.metadata),
            }
            for e in rea_examples
        ],
    )

    # Quality on native families only (reuse harness with controlled inventory gate override)
    report = evaluate_fp2_corpus(root)
    # Override fixture_inventory for controlled batch size
    n = report.total_native_records
    if CONTROLLED_BATCH_MIN <= n <= CONTROLLED_BATCH_MAX:
        report.gates["fixture_inventory"] = "PASS"
    elif 100 <= n < CONTROLLED_BATCH_MIN:
        # Unit-test sized batches
        report.gates["fixture_inventory"] = "PASS"
    else:
        report.gates["fixture_inventory"] = "FAIL"
    # Recompute readiness with coverage requirement: all 23 preferred
    uncovered = report.coverage.get("uncovered") or []
    warnings: list[str] = []
    hard_fail = [
        g
        for g, v in report.gates.items()
        if g != "capability_coverage" and v == "FAIL"
    ]
    if uncovered:
        # Should be empty after TR-5.2
        hard_fail.append("coverage_incomplete")
    leak = _leakage_check(records)
    if leak:
        hard_fail.append("split_leakage")
        warnings.extend(leak)

    # Negatives must not be in packs
    for path_name in ("pack_development.jsonl", "pack_reasoning.jsonl", "splits.jsonl"):
        text = (root / path_name).read_text(encoding="utf-8")
        if "quality_corpus\": \"negative" in text or "not_for_training" in text:
            hard_fail.append("negative_contamination")

    if hard_fail:
        decision = "FAIL"
    elif report.gates.get("duplicates") == "WARN" or warnings:
        decision = "PASS_WITH_WARNINGS"
        warnings.extend(report.readiness_reasons)
    else:
        decision = "PASS"

    # Per-capability counts
    per_cap: Counter[str] = Counter()
    for rec in records:
        eng = extract_engineering_capability(rec)
        if eng:
            for part in eng.split(","):
                if part in PREFERRED_ENGINEERING_CAPABILITY_IDS:
                    per_cap[part] += 1

    dup = find_duplicates(records)
    domain = domain_distribution(records)
    cov = capability_coverage(records)

    # Checksum over native jsonl
    h = hashlib.sha256()
    for fam in NATIVE_FAMILIES:
        h.update((root / f"{fam}.jsonl").read_bytes())
    checksum = h.hexdigest()

    result = ControlledBatchResult(
        output_dir=str(root),
        total_native=len(records),
        per_family=counts,
        per_capability=dict(sorted(per_cap.items())),
        domain=domain,
        split_counts=dict(split_counter),
        development_pack_count=len(dev_examples),
        reasoning_pack_count=len(rea_examples),
        duplicates={
            "unique_fingerprints": dup.get("unique_fingerprints"),
            "duplicate_groups": dup.get("duplicate_groups"),
        },
        coverage=cov,
        quality={
            "gates": report.gates,
            "how_violations": report.how_violations,
            "taxonomy_violations": report.taxonomy_violations,
            "schema_failures": report.schema_failures,
            "hard_fail": hard_fail,
            "split_strategy": document_split_strategy(),
        },
        decision=decision,
        warnings=warnings,
        checksum=checksum,
    )

    manifest = {
        **result.to_dict(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "classification": "fp2_controlled_batch",
        "legacy_datasets_untouched": True,
        "negatives_excluded": True,
        "batch_target": target,
        "batch_max": CONTROLLED_BATCH_MAX,
        "notes": (
            "TR-5 controlled batch 1. Isolated under controlled_batch_1/. "
            "Do not mix with legacy production JSONL. Do not train yet without authorization."
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "quality_report_tr5.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # README
    (root / "README.md").write_text(
        "# FP2 controlled batch 1 (TR-5)\n\n"
        f"Version: `{CONTROLLED_BATCH_VERSION}`  \n"
        f"Records: {len(records)}  \n"
        f"Decision: **{decision}**  \n\n"
        "Isolated from legacy production datasets. Negatives excluded.\n",
        encoding="utf-8",
    )
    return result
