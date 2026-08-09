"""TR-6 evaluation helpers — diversity, packs, continuity, objective semantics."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aiodoo_training.system_training_contract.forbidden import FORBIDDEN_IMPL_IDS
from aiodoo_training.system_training_contract.quality.common import (
    NATIVE_FAMILIES,
    extract_engineering_capability,
    load_jsonl,
    model_facing_blob,
)
from aiodoo_training.system_training_contract.quality.gates import (
    scan_forbidden_how,
    scan_taxonomy,
)
from aiodoo_training.system_training_contract.quality.splits import assign_split, scenario_key
from aiodoo_training.system_training_contract.taxonomy import (
    DEVELOPMENT_PROVIDER_CAPABILITIES,
    PREFERRED_ENGINEERING_CAPABILITY_IDS,
    PROVIDER_CAPABILITY_IDS,
    REASONING_PROVIDER_CAPABILITIES,
)

_ROUND_SUFFIX = re.compile(r"_r\d+$")
_CONTEXT_SUFFIX = re.compile(r"\s*\(context \d+\)\s*$", re.I)

_HOW_PACK_TOKENS = frozenset(
    {
        *FORBIDDEN_IMPL_IDS,
        "strategy",
        "resolver",
        "implementationframework",
        "pytest",
        "mypy",
        "pyright",
        "adapters_required",
    }
)


def normalize_scenario_family(family: str) -> str:
    """Collapse round/context variants into one semantic family."""
    fam = (family or "").strip()
    fam = _ROUND_SUFFIX.sub("", fam)
    return fam or "unknown"


def load_native_records(corpus_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fam in NATIVE_FAMILIES:
        rows.extend(load_jsonl(corpus_root / f"{fam}.jsonl"))
    return rows


def load_pack(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def scenario_family_of(rec: Mapping[str, Any]) -> str:
    meta = rec.get("metadata") if isinstance(rec.get("metadata"), Mapping) else {}
    raw = str(meta.get("scenario_family") or meta.get("scenario") or scenario_key(rec))
    return normalize_scenario_family(raw)


def analyze_scenario_diversity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for rec in records:
        counts[scenario_family_of(rec)] += 1
    total = max(1, len(records))
    if not counts:
        return {
            "unique_families": 0,
            "largest_family": None,
            "largest_family_count": 0,
            "concentration_pct": 0.0,
            "top_families": {},
            "repetitive_families": [],
            "records_per_family_avg": 0.0,
        }
    largest, largest_n = counts.most_common(1)[0]
    repetitive = [f for f, n in counts.items() if n >= max(40, int(0.05 * total))]
    return {
        "unique_families": len(counts),
        "largest_family": largest,
        "largest_family_count": largest_n,
        "concentration_pct": round(100.0 * largest_n / total, 2),
        "top_families": dict(counts.most_common(15)),
        "repetitive_families": sorted(repetitive),
        "records_per_family_avg": round(total / max(1, len(counts)), 2),
        "family_counts": dict(sorted(counts.items())),
    }


def analyze_capability_distribution(
    records: Sequence[Mapping[str, Any]],
    *,
    pack_dev: Sequence[Mapping[str, Any]] | None = None,
    pack_rea: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    matrix: dict[str, dict[str, Any]] = {
        cap: {
            "count": 0,
            "families": set(),
            "successish": 0,
            "failureish": 0,
            "odoo": 0,
            "generic": 0,
            "development": 0,
            "reasoning": 0,
        }
        for cap in sorted(PREFERRED_ENGINEERING_CAPABILITY_IDS)
    }

    def _bump(cap: str, rec: Mapping[str, Any], *, plane: str | None = None) -> None:
        if cap not in matrix:
            return
        matrix[cap]["count"] += 1
        matrix[cap]["families"].add(str(rec.get("record_type") or ""))
        if rec.get("domain_specialization") == "odoo":
            matrix[cap]["odoo"] += 1
        else:
            matrix[cap]["generic"] += 1
        status = _status_hint(rec)
        if status == "success":
            matrix[cap]["successish"] += 1
        elif status == "failure":
            matrix[cap]["failureish"] += 1
        if plane == "development":
            matrix[cap]["development"] += 1
        elif plane == "reasoning":
            matrix[cap]["reasoning"] += 1

    for rec in records:
        eng = extract_engineering_capability(rec)
        if not eng:
            continue
        for part in eng.split(","):
            _bump(part.strip(), rec)

    # Pack-level capability mentions from assistant JSON
    for plane, pack in (("development", pack_dev or ()), ("reasoning", pack_rea or ())):
        for ex in pack:
            caps = _caps_from_example(ex)
            for cap in caps:
                if cap in matrix:
                    matrix[cap][plane] += 1

    serial = {
        c: {
            **{k: v for k, v in vals.items() if k != "families"},
            "families": sorted(vals["families"]),
        }
        for c, vals in matrix.items()
    }
    under = [c for c, v in serial.items() if v["count"] < 20]
    over = [c for c, v in serial.items() if v["count"] >= 55]
    superficial = [
        c
        for c, v in serial.items()
        if v["count"] > 0 and len(v["families"]) <= 1 and v["count"] < 25
    ]
    return {
        "matrix": serial,
        "underrepresented": under,
        "overrepresented": over,
        "superficial": superficial,
    }


def _status_hint(rec: Mapping[str, Any]) -> str:
    evidence = rec.get("evidence") if isinstance(rec.get("evidence"), Mapping) else {}
    expected = rec.get("expected_output") if isinstance(rec.get("expected_output"), Mapping) else {}
    status = str(
        evidence.get("status")
        or evidence.get("objective_state")
        or expected.get("decision_kind")
        or ""
    ).lower()
    if status in {"succeeded", "complete", "passed", "continue"}:
        return "success"
    if status in {"failed", "failure", "blocked", "escalated", "cancelled", "replan", "escalate"}:
        return "failure"
    return "other"


def _caps_from_example(ex: Mapping[str, Any]) -> set[str]:
    caps: set[str] = set()
    messages = ex.get("messages") or ()
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        content = str(msg.get("content") or "")
        try:
            # assistant is usually pure JSON; user has preamble + JSON
            blob = content
            if "\n\n{" in content:
                blob = content.split("\n\n", 1)[1]
            data = json.loads(blob)
        except Exception:
            continue
        if isinstance(data, Mapping):
            cid = data.get("capability_id")
            if cid:
                caps.add(str(cid))
            for step in data.get("steps") or ():
                if isinstance(step, Mapping) and step.get("action"):
                    caps.add(str(step["action"]))
            ev = data.get("evidence")
            if isinstance(ev, Mapping) and ev.get("capability_id"):
                caps.add(str(ev["capability_id"]))
    return caps


def analyze_packs(pack_dev: Sequence[Mapping[str, Any]], pack_rea: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []

    def _check_pack(name: str, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        local_issues: list[str] = []
        by_type: Counter[str] = Counter()
        by_provider: Counter[str] = Counter()
        how_hits = 0
        malformed = 0
        trivial = 0
        for ex in examples:
            if not isinstance(ex, Mapping):
                malformed += 1
                continue
            if "example_id" not in ex or "messages" not in ex or "dataset_type" not in ex:
                malformed += 1
                local_issues.append(f"{name}:missing_fields:{ex.get('example_id')}")
                continue
            msgs = ex.get("messages") or ()
            if len(msgs) < 2:
                malformed += 1
                continue
            roles = [m.get("role") for m in msgs if isinstance(m, Mapping)]
            if roles[:2] != ["user", "assistant"]:
                malformed += 1
                local_issues.append(f"{name}:bad_roles:{ex.get('example_id')}")
            user = str(msgs[0].get("content") or "") if isinstance(msgs[0], Mapping) else ""
            asst = str(msgs[1].get("content") or "") if isinstance(msgs[1], Mapping) else ""
            # Scan assistant fully; for user, scan only JSON payload after preamble.
            user_payload = user
            if "\n\n" in user:
                user_payload = user.split("\n\n", 1)[1]
            scan_blob = (user_payload + "\n" + asst).lower()
            for tok in _HOW_PACK_TOKENS:
                if tok in scan_blob:
                    how_hits += 1
                    local_issues.append(f"{name}:how:{ex.get('example_id')}:{tok}")
                    break
            # Assistant must be JSON
            try:
                payload = json.loads(asst)
                if not isinstance(payload, Mapping):
                    malformed += 1
            except Exception:
                malformed += 1
                local_issues.append(f"{name}:assistant_not_json:{ex.get('example_id')}")
                payload = {}
            meta = ex.get("metadata") if isinstance(ex.get("metadata"), Mapping) else {}
            rtype = str(meta.get("record_type") or payload.get("record_type") or "")
            if rtype:
                by_type[rtype] += 1
            ds = str(ex.get("dataset_type") or "")
            by_provider[ds] += 1
            # Trivial: empty assistant or only decision_kind without reason for loop
            if isinstance(payload, Mapping):
                if payload.get("decision_kind") in {"replan", "complete", "escalate"} and not str(
                    payload.get("reason") or ""
                ).strip():
                    trivial += 1
                if payload.get("capability_id") and not payload.get("args") and rtype == "capability_intent":
                    # args may be empty legitimately
                    pass
        return {
            "count": len(examples),
            "malformed": malformed,
            "how_hits": how_hits,
            "trivial_flags": trivial,
            "by_record_type": dict(by_type),
            "by_dataset_type": dict(by_provider),
            "issues": local_issues[:50],
            "issue_count": len(local_issues),
        }

    return {
        "development": _check_pack("development", pack_dev),
        "reasoning": _check_pack("reasoning", pack_rea),
    }


def analyze_objective_completion(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Metrics: operation success ≠ automatic COMPLETE; failure ≠ automatic repair."""
    feedback = [r for r in records if r.get("record_type") == "engineering_feedback"]
    loops = [r for r in records if r.get("record_type") == "loop_decision"]
    plans = [r for r in records if r.get("record_type") == "planning_decision"]

    op_ok_obj_incomplete = 0
    op_fail_not_repair = 0
    auto_repair_policy = 0
    auto_validate_after_repair = 0
    complete_without_success_evidence = 0

    for r in feedback:
        ev = r.get("evidence") if isinstance(r.get("evidence"), Mapping) else {}
        if ev.get("execution_state") == "succeeded" and ev.get("objective_state") in {
            "incomplete",
            "in_progress",
            "blocked",
        }:
            op_ok_obj_incomplete += 1
        if ev.get("execution_state") == "failed":
            # recommended continuation should not always be a forced repair capability
            cont = str(ev.get("recommended_continuation") or "").lower()
            if cont not in {"repair", "execution.repair"}:
                op_fail_not_repair += 1

    for r in loops:
        out = r.get("expected_output") if isinstance(r.get("expected_output"), Mapping) else {}
        reason = str(out.get("reason") or "").lower()
        if "automatically repair" in reason or "must repair" in reason:
            auto_repair_policy += 1
        if "must validate after repair" in reason or "always validate after repair" in reason:
            auto_validate_after_repair += 1
        if out.get("decision_kind") == "complete" and "validation passed" not in reason and "objective complete" not in reason:
            # soft: complete reasons should cite objective/evidence
            if "blocker" in reason:
                complete_without_success_evidence += 1

    for r in plans:
        out = r.get("expected_output") if isinstance(r.get("expected_output"), Mapping) else {}
        steps = out.get("steps") or ()
        actions = [
            str(s.get("action") or "")
            for s in steps
            if isinstance(s, Mapping)
        ]
        # Detect hard-coded validation failure → repair pairs as sole two-step policy
        if actions == ["execution.repair", "validation.run"]:
            # Allowed as one scenario among many, but count as potential bias if dominant
            auto_validate_after_repair += 1

    return {
        "operation_success_objective_incomplete": op_ok_obj_incomplete,
        "operation_failure_not_forced_repair": op_fail_not_repair,
        "auto_repair_policy_reasons": auto_repair_policy,
        "repair_then_validate_plan_pairs": auto_validate_after_repair,
        "ok": (
            op_ok_obj_incomplete >= 1
            and auto_repair_policy == 0
            and op_fail_not_repair >= 1
        ),
    }


def analyze_continuity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    states = [r for r in records if r.get("record_type") == "engineering_state"]
    dcs = [r for r in records if r.get("record_type") == "decision_context"]
    issues: list[str] = []
    scenarios = {
        str((r.get("metadata") or {}).get("scenario") or ""): r for r in states
    }
    for name in (
        "cycle1_validation_failed",
        "cycle2_repair_applied",
        "cycle3_validation_passed",
        "previous_complete_current_failure",
        "previous_failure_current_success",
    ):
        if name not in scenarios:
            # controlled batch may use scenario_family narrative form
            found = any(name in str((r.get("metadata") or {})) for r in states)
            if not found and name.startswith("cycle"):
                # narrative_partner_a cycles
                pass
            elif not found:
                issues.append(f"missing_state_scenario:{name}")

    # Narrative partner cycles
    narr = [
        r
        for r in states
        if str((r.get("metadata") or {}).get("scenario_family") or "").startswith("narrative_partner")
    ]
    if narr:
        by_cycle = {
            int((r.get("evidence") or {}).get("cycle_index") or -1): r for r in narr
        }
        if 1 in by_cycle and by_cycle[1]["evidence"].get("objective_state") not in {
            "incomplete",
            "failed",
        }:
            issues.append("narrative_cycle1_not_incomplete")
        if 3 in by_cycle and by_cycle[3]["evidence"].get("objective_state") != "complete":
            issues.append("narrative_cycle3_not_complete")

    for dc in dcs:
        inp = dc.get("input") if isinstance(dc.get("input"), Mapping) else {}
        blob = json.dumps(inp).lower()
        for tok in ("local_workspace", "implementation_id", "stdout", "password", "command"):
            if tok in blob:
                issues.append(f"dc_how:{dc.get('record_id')}:{tok}")
        hist = inp.get("bounded_history") or ()
        if not isinstance(hist, (list, tuple)):
            issues.append(f"dc_bad_history:{dc.get('record_id')}")
        # current vs historical: if history says complete and current failed — good isolation
        if inp.get("objective_state") == "failed":
            for h in hist:
                if isinstance(h, Mapping) and h.get("objective_state") == "complete" and h.get("historical"):
                    break

    return {"ok": not issues, "issues": issues, "state_count": len(states), "dc_count": len(dcs)}


def analyze_odoo_quality(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """TR-7 semantic domain quality (replaces TR-6 narrow-token heuristic).

    TR-6 counted as ambiguous any ``domain_specialization=odoo`` record lacking
    literal ``odoo``/``manifest``/``addons/`` tokens — falsely flagging valid
    cues such as ``models/partner.py`` / ``action_confirm``. TR-7 classifies
    from semantic content via ``domain_classify``.
    """
    from aiodoo_training.system_training_contract.evaluation.domain_classify import (
        DomainClass,
        classify_domain,
        has_odoo_semantics,
    )

    generic = odoo = ambiguous = 0
    ambiguous_ids: list[str] = []
    for rec in records:
        cls = classify_domain(rec)
        if cls == DomainClass.ODOO_SPECIALIZED:
            odoo += 1
        elif cls == DomainClass.GENERIC:
            generic += 1
        else:
            ambiguous += 1
            ambiguous_ids.append(str(rec.get("record_id")))
    return {
        "generic": generic,
        "odoo": odoo,
        "ambiguous": ambiguous,
        "ambiguous_ids_sample": ambiguous_ids[:20],
        "odoo_pct": round(100.0 * odoo / max(1, generic + odoo), 2),
        "classifier": "tr7_semantic_cues",
        "odoo_semantics_true_positives": sum(
            1 for r in records if r.get("domain_specialization") == "odoo" and has_odoo_semantics(r)
        ),
    }


def analyze_splits(records: Sequence[Mapping[str, Any]], splits_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    id_to_split = {str(r.get("record_id")): str(r.get("split")) for r in splits_rows}
    for rec in records:
        rid = str(rec.get("record_id"))
        sp = id_to_split.get(rid) or assign_split(rec).value
        by_split[sp].append(rec)

    # leakage: narrative families across splits
    fam_splits: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        fam = scenario_family_of(rec)
        rid = str(rec.get("record_id"))
        sp = id_to_split.get(rid) or assign_split(rec).value
        fam_splits[fam].add(sp)
    leaks = {f: sorted(s) for f, s in fam_splits.items() if len(s) > 1 and f.startswith("narrative_")}

    # capability presence per split
    cap_splits: dict[str, set[str]] = defaultdict(set)
    for sp, rows in by_split.items():
        for rec in rows:
            eng = extract_engineering_capability(rec)
            if not eng:
                continue
            for part in eng.split(","):
                if part in PREFERRED_ENGINEERING_CAPABILITY_IDS:
                    cap_splits[part].add(sp)
    train_only = [c for c, sps in cap_splits.items() if sps == {"train"}]
    test_only = [c for c, sps in cap_splits.items() if sps == {"test"}]

    counts = {k: len(v) for k, v in by_split.items()}
    return {
        "counts": counts,
        "narrative_leakage": leaks,
        "leakage_count": len(leaks),
        "capabilities_train_only": train_only,
        "capabilities_test_only": test_only,
        "ok": len(leaks) == 0,
    }


def analyze_negative_contamination(corpus_root: Path, pack_dev: Sequence[Mapping], pack_rea: Sequence[Mapping]) -> dict[str, Any]:
    issues: list[str] = []
    for name in (
        "pack_development.jsonl",
        "pack_reasoning.jsonl",
        "splits.jsonl",
        *[f"{f}.jsonl" for f in NATIVE_FAMILIES],
    ):
        path = corpus_root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "not_for_training" in text or 'quality_corpus": "negative' in text:
            issues.append(f"contaminated:{name}")
    parent_neg = corpus_root.parent / "quality_negatives.jsonl"
    neg_ok = False
    if parent_neg.is_file():
        for line in parent_neg.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("not_for_training") is True or row.get("quality_only") is True:
                neg_ok = True
                break
            # nested flag
            if row.get("record", {}).get("metadata", {}).get("quality_corpus", "").startswith("negative"):
                neg_ok = True
                break
    for ex in list(pack_dev) + list(pack_rea):
        meta = ex.get("metadata") if isinstance(ex, Mapping) else {}
        if isinstance(meta, Mapping) and (
            meta.get("not_for_training") or str(meta.get("quality_corpus") or "").startswith("negative")
        ):
            issues.append(f"pack_negative:{ex.get('example_id')}")
    return {"ok": not issues and neg_ok, "issues": issues, "negatives_file_present": parent_neg.is_file()}


def analyze_provider_separation(records: Sequence[Mapping[str, Any]], packs: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    for rec in records:
        provider = str(rec.get("provider_capability") or "")
        if provider and provider not in PROVIDER_CAPABILITY_IDS:
            issues.append(f"bad_provider:{rec.get('record_id')}:{provider}")
        eng = extract_engineering_capability(rec)
        if eng:
            for part in eng.split(","):
                if part in PROVIDER_CAPABILITY_IDS:
                    issues.append(f"provider_as_eng:{rec.get('record_id')}:{part}")
        for issue in scan_taxonomy(rec):
            issues.append(f"{rec.get('record_id')}:{issue}")
    # Pack dataset types should be provider names or mixed
    for plane, key in (("development", "development"), ("reasoning", "reasoning")):
        by_ds = (packs.get(key) or {}).get("by_dataset_type") or {}
        for ds in by_ds:
            if ds not in PROVIDER_CAPABILITY_IDS and ds != "mixed":
                issues.append(f"pack_dataset_type:{plane}:{ds}")
    return {
        "ok": not issues,
        "issues": issues[:40],
        "development_providers": sorted(DEVELOPMENT_PROVIDER_CAPABILITIES),
        "reasoning_providers": sorted(REASONING_PROVIDER_CAPABILITIES),
    }
