"""TR-7 controlled batch derivative — continuity expansion + domain cleanup.

Reads TR-5 ``controlled_batch_1`` (immutable evidence), writes versioned
``controlled_batch_2`` with corrected domain labels and expanded Reasoning
continuity. Does not overwrite TR-5. Does not train adapters.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiodoo_training.system_training_contract.evaluation.domain_classify import (
    DomainClass,
    classify_domain,
    corrected_domain_specialization,
)
from aiodoo_training.system_training_contract.evaluation.harness import (
    evaluate_controlled_batch,
    write_tr6_report,
)
from aiodoo_training.system_training_contract.evaluation.metrics import load_native_records
from aiodoo_training.system_training_contract.evaluation.scorecard import render_tr6_scorecard
from aiodoo_training.system_training_contract.generators.common import write_jsonl
from aiodoo_training.system_training_contract.generators.tr7_continuity import (
    TR7_BATCH_VERSION,
    generate_continuity_expansion,
)
from aiodoo_training.system_training_contract.quality.common import NATIVE_FAMILIES
from aiodoo_training.system_training_contract.quality.formatters import format_fp2_pack
from aiodoo_training.system_training_contract.quality.splits import assign_split, scenario_key
from aiodoo_training.system_training_contract.records import (
    TrainingRecordError,
    validate_record_mapping,
)
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

TR5_SOURCE_VERSION = "fp2-controlled-1.0.0"
TR7_SOURCE_BATCH = "controlled_batch_1"
TR7_OUTPUT_BATCH = "controlled_batch_2"


@dataclass
class Tr7EmitResult:
    output_dir: Path
    source_dir: Path
    total_native: int
    continuity_added: int
    domain_actions: dict[str, int] = field(default_factory=dict)
    quarantined: int = 0
    per_family: dict[str, int] = field(default_factory=dict)
    development_pack: int = 0
    reasoning_pack: int = 0
    splits: dict[str, int] = field(default_factory=dict)
    readiness: str = ""
    report_path: Path | None = None


def _apply_domain_correction(rec: Mapping[str, Any]) -> tuple[dict[str, Any], str, DomainClass]:
    """Return corrected record, action, and resulting class."""
    out = copy.deepcopy(dict(rec))
    new_dom, cls, action = corrected_domain_specialization(out)
    old_dom = out.get("domain_specialization")
    out["domain_specialization"] = new_dom
    meta = out.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        out["metadata"] = meta
    meta = dict(meta)
    meta["tr7_domain_action"] = action
    meta["tr7_domain_class"] = cls.value
    meta["tr7_source_batch"] = TR7_SOURCE_BATCH
    meta["tr7_source_record_id"] = out.get("record_id")
    if action != "keep":
        meta["tr7_domain_previous"] = old_dom
        meta["controlled_batch"] = TR7_BATCH_VERSION
    out["metadata"] = meta
    # Also mirror into evidence/current_fields if present as mirror fields — no;
    # domain_specialization is top-level only.
    return out, action, cls


def _quarantine_record(rec: Mapping[str, Any], reason: str) -> dict[str, Any]:
    out = copy.deepcopy(dict(rec))
    meta = dict(out.get("metadata") or {})
    meta["quality_corpus"] = "ambiguous_quarantine"
    meta["not_for_training"] = True
    meta["tr7_quarantine_reason"] = reason
    meta["tr7_source_batch"] = TR7_SOURCE_BATCH
    out["metadata"] = meta
    return out


def build_tr7_records(source_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Correct domains; quarantine unprovable ambiguous; append continuity."""
    actions: Counter[str] = Counter()
    training: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []

    for rec in source_records:
        corrected, action, cls = _apply_domain_correction(rec)
        actions[action] += 1
        # After correction, re-classify — should never be AMBIGUOUS for corrected_domain path
        post = classify_domain(corrected)
        if post == DomainClass.AMBIGUOUS:
            actions["quarantine"] += 1
            quarantined.append(
                _quarantine_record(corrected, "unprovable_domain_classification")
            )
            continue
        training.append(corrected)

    continuity = generate_continuity_expansion()
    for rec in continuity:
        validate_record_mapping(rec)
        # Continuity records already have domain set correctly when cues present
        corrected, action, cls = _apply_domain_correction(rec)
        actions[f"continuity_{action}"] += 1
        if classify_domain(corrected) == DomainClass.AMBIGUOUS:
            quarantined.append(
                _quarantine_record(corrected, "continuity_ambiguous_domain")
            )
            continue
        meta = dict(corrected.get("metadata") or {})
        meta["tr7_continuity"] = True
        meta["controlled_batch"] = TR7_BATCH_VERSION
        corrected["metadata"] = meta
        training.append(corrected)

    actions["continuity_added"] = len(continuity)
    return training, quarantined, dict(actions)


def emit_tr7_batch(
    *,
    source_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Tr7EmitResult:
    # Resolve sibling datasets repo from this package (cwd-independent).
    workspace = Path(__file__).resolve().parents[4]
    datasets = workspace / "aiodoo-datasets" / "datasets" / "fp2"
    src = Path(source_dir) if source_dir else datasets / TR7_SOURCE_BATCH
    out = Path(output_dir) if output_dir else datasets / TR7_OUTPUT_BATCH
    out.mkdir(parents=True, exist_ok=True)

    source_records = load_native_records(src)
    if len(source_records) != 1200:
        raise RuntimeError(f"TR-5 source expected 1200 records, got {len(source_records)}")

    training, quarantined, actions = build_tr7_records(source_records)

    # Validate all training records
    for rec in training:
        try:
            validate_record_mapping(rec)
        except TrainingRecordError as exc:
            raise RuntimeError(f"invalid TR-7 record {rec.get('record_id')}: {exc}") from exc

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in training:
        by_family[str(rec.get("record_type") or "unknown")].append(rec)

    counts: dict[str, int] = {}
    for fam in NATIVE_FAMILIES:
        counts[fam] = write_jsonl(out / f"{fam}.jsonl", by_family.get(fam, []))

    write_jsonl(out / "ambiguous_quarantine.jsonl", quarantined)
    # Carry projection fixtures from source for audit continuity (not training packs)
    proj_src = src / "projection_fixtures.jsonl"
    if proj_src.is_file():
        (out / "projection_fixtures.jsonl").write_bytes(proj_src.read_bytes())

    # Audit log of domain corrections (traceability; not for training)
    audit_rows = []
    for rec in training:
        meta = rec.get("metadata") or {}
        if meta.get("tr7_domain_action") and meta.get("tr7_domain_action") != "keep":
            audit_rows.append(
                {
                    "record_id": rec.get("record_id"),
                    "record_type": rec.get("record_type"),
                    "action": meta.get("tr7_domain_action"),
                    "previous": meta.get("tr7_domain_previous"),
                    "domain_specialization": rec.get("domain_specialization"),
                    "class": meta.get("tr7_domain_class"),
                }
            )
    write_jsonl(out / "tr7_domain_audit.jsonl", audit_rows)

    # Splits (same fp2-split-1.0.0 algorithm)
    split_rows = []
    split_counter: Counter[str] = Counter()
    for rec in training:
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
    write_jsonl(out / "splits.jsonl", split_rows)

    # Packs — exclude quarantine by construction (not in training list)
    # Guard: never include not_for_training
    for rec in training:
        meta = rec.get("metadata") or {}
        if meta.get("not_for_training") or meta.get("quality_corpus") == "negative":
            raise RuntimeError(f"training contamination: {rec.get('record_id')}")

    dev_examples = format_fp2_pack(training, pack="development")
    rea_examples = format_fp2_pack(training, pack="reasoning")
    write_jsonl(
        out / "pack_development.jsonl",
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
        out / "pack_reasoning.jsonl",
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

    h = hashlib.sha256()
    for fam in NATIVE_FAMILIES:
        h.update((out / f"{fam}.jsonl").read_bytes())
    checksum = h.hexdigest()

    cont_n = sum(1 for r in training if (r.get("metadata") or {}).get("tr7_continuity"))
    manifest = {
        "controlled_batch_version": TR7_BATCH_VERSION,
        "version": TR7_BATCH_VERSION,
        "source_batch": TR7_SOURCE_BATCH,
        "source_version": TR5_SOURCE_VERSION,
        "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_native": len(training),
        "continuity_added": cont_n,
        "domain_actions": actions,
        "quarantined": len(quarantined),
        "per_family": counts,
        "development_pack": len(dev_examples),
        "reasoning_pack": len(rea_examples),
        "splits": dict(split_counter),
        "checksum": checksum,
        "tr7": True,
        "notes": [
            "TR-5 controlled_batch_1 preserved immutable",
            "Domain labels corrected via semantic cues; unjustified odoo cleared",
            "Ambiguous quarantined outside packs/splits",
            "Continuity expanded with multi-cycle narratives",
            "Negatives remain in parent quality_negatives.jsonl",
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = evaluate_controlled_batch(out)
    report_path = out / "quality_report_tr7.json"
    write_tr6_report(report, report_path)
    (out / "scorecard_tr7.md").write_text(render_tr6_scorecard(report) + "\n", encoding="utf-8")

    # Also mirror evaluation as tr6-compatible name for harness consumers
    write_tr6_report(report, out / "quality_report_tr6.json")

    return Tr7EmitResult(
        output_dir=out,
        source_dir=src,
        total_native=len(training),
        continuity_added=cont_n,
        domain_actions=actions,
        quarantined=len(quarantined),
        per_family=counts,
        development_pack=len(dev_examples),
        reasoning_pack=len(rea_examples),
        splits=dict(split_counter),
        readiness=report.readiness,
        report_path=report_path,
    )


def main() -> None:
    result = emit_tr7_batch()
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "total_native": result.total_native,
                "continuity_added": result.continuity_added,
                "domain_actions": result.domain_actions,
                "quarantined": result.quarantined,
                "per_family": result.per_family,
                "development_pack": result.development_pack,
                "reasoning_pack": result.reasoning_pack,
                "splits": result.splits,
                "readiness": result.readiness,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
