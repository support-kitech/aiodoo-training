"""AT-7.3 — Emit Reasoning sparse-skill controlled corpora.

Conversation + Approval: generate, analyze, write fixtures/datasets.
Evaluation: semantics report only (EVALUATION_SEMANTICS_UNRESOLVED).

Does NOT modify controlled_batch_2. Does NOT train adapters.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aiodoo_training.system_training_contract.generators.approval_controlled import (
    APPROVAL_CONTROLLED_GENERATOR,
    MIN_FAMILIES as APPROVAL_MIN_FAMILIES,
    REASONING_SPARSE_VERSION,
    TARGET_MAX as APPROVAL_TARGET_MAX,
    TARGET_MIN as APPROVAL_TARGET_MIN,
    approval_family_count,
    generate_approval_controlled_records,
)
from aiodoo_training.system_training_contract.generators.common import write_jsonl
from aiodoo_training.system_training_contract.generators.conversation_controlled import (
    CONVERSATION_CONTROLLED_GENERATOR,
    MIN_FAMILIES as CONVERSATION_MIN_FAMILIES,
    TARGET_MAX as CONVERSATION_TARGET_MAX,
    TARGET_MIN as CONVERSATION_TARGET_MIN,
    conversation_family_count,
    generate_conversation_controlled_records,
)
from aiodoo_training.system_training_contract.generators.evaluation_semantics import (
    EVALUATION_SEMANTIC_DEFINITION,
)
from aiodoo_training.system_training_contract.quality.analysis import (
    domain_distribution,
    find_duplicates,
)
from aiodoo_training.system_training_contract.quality.common import stable_dumps
from aiodoo_training.system_training_contract.quality.formatters import format_fp2_pack
from aiodoo_training.system_training_contract.quality.gates import (
    scan_forbidden_how,
    scan_taxonomy,
)
from aiodoo_training.system_training_contract.quality.negatives import (
    REASONING_SPARSE_NEGATIVE_CASES,
    evaluate_negative_case,
)
from aiodoo_training.system_training_contract.quality.splits import (
    assign_split,
    document_split_strategy,
    scenario_key,
)
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT.parent
_DEFAULT_TRAINING = _REPO_ROOT / "fixtures" / "fp2" / "reasoning_controlled_1"
_DEFAULT_DATASETS = (
    _WORKSPACE_ROOT / "aiodoo-datasets" / "datasets" / "fp2" / "reasoning_controlled_1"
)
BATCH2_CHECKSUM = "728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227"
BATCH2 = _WORKSPACE_ROOT / "aiodoo-datasets" / "datasets" / "fp2" / "controlled_batch_2"


def normalized_fingerprint(record: Mapping[str, Any]) -> str:
    blob = stable_dumps(
        {
            "record_type": record.get("record_type"),
            "provider": record.get("provider_capability"),
            "domain": record.get("domain_specialization") or "generic",
            "input": record.get("input"),
            "expected_output": record.get("expected_output"),
            "evidence": record.get("evidence"),
        }
    ).lower()
    blob = re.sub(r"[a-z0-9_./\\-]+\.(py|xml|csv|md|html|json)", "<PATH>", blob)
    blob = re.sub(r"\b\d+\b", "<N>", blob)
    blob = re.sub(r"\s+", " ", blob)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def find_normalized_duplicates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for r in records:
        buckets[normalized_fingerprint(r)].append(str(r.get("record_id") or "?"))
    dups = {fp: ids for fp, ids in buckets.items() if len(ids) > 1}
    return {
        "unique_normalized_fingerprints": len(buckets),
        "normalized_duplicate_groups": len(dups),
        "normalized_duplicate_record_ids": dups,
    }


def _analyze_capability(
    records: Sequence[Mapping[str, Any]],
    *,
    provider: str,
    target_min: int,
    target_max: int,
    min_families: int,
    ready_verdict: str,
    needs_fix_verdict: str,
) -> dict[str, Any]:
    by_type = Counter(str(r["record_type"]) for r in records)
    families = Counter(
        str((r.get("metadata") or {}).get("scenario_family") or "?") for r in records
    )
    domain = domain_distribution(list(records))
    exact = find_duplicates(list(records))
    normalized = find_normalized_duplicates(records)

    split_counts: Counter[str] = Counter()
    family_splits: dict[str, set[str]] = defaultdict(set)
    split_rows: list[dict[str, Any]] = []
    for r in records:
        split = assign_split(r).value
        fam = str((r.get("metadata") or {}).get("scenario_family") or "")
        split_counts[split] += 1
        family_splits[fam].add(split)
        split_rows.append(
            {
                "record_id": r.get("record_id"),
                "split": split,
                "scenario_family": fam,
                "scenario_key": scenario_key(r),
                "record_type": r.get("record_type"),
            }
        )
    leakage = {
        fam: sorted(splits)
        for fam, splits in sorted(family_splits.items())
        if len(splits) > 1
    }

    how_hits = sum(1 for r in records if scan_forbidden_how(r))
    tax_hits = sum(1 for r in records if scan_taxonomy(r))
    legacy_hits = sum(
        1
        for r in records
        if r.get("metadata", {}).get("legacy")
        or "context_v1_0" in json.dumps(r)
        or r.get("provider_capability") != provider
    )
    neg_contam = sum(
        1
        for r in records
        if str((r.get("metadata") or {}).get("quality_corpus") or "").startswith("negative")
    )
    neg_results = [evaluate_negative_case(c) for c in REASONING_SPARSE_NEGATIVE_CASES]
    neg_ok = all(r["matched"] for r in neg_results)

    pack = format_fp2_pack(list(records), pack="reasoning")
    pack_ok = len(pack) == len(records) and all(
        ex.dataset_type.value == provider for ex in pack
    )
    provider_dataset_equiv = pack_ok

    decision_kinds = Counter(
        str((r.get("expected_output") or {}).get("decision_kind") or "")
        for r in records
        if r.get("record_type") == "loop_decision"
    )

    hard_fail = False
    reasons: list[str] = []
    if not (target_min <= len(records) <= target_max):
        hard_fail = True
        reasons.append("count_out_of_band")
    if len(families) < min_families:
        hard_fail = True
        reasons.append("insufficient_families")
    if exact["duplicate_groups"] != 0:
        hard_fail = True
        reasons.append("exact_duplicates")
    if how_hits or tax_hits or legacy_hits or neg_contam:
        hard_fail = True
        reasons.append("safety_contamination")
    if leakage:
        hard_fail = True
        reasons.append("family_leakage")
    if not pack_ok:
        hard_fail = True
        reasons.append("pack_invalid")
    if split_counts.get("validation", 0) == 0 or split_counts.get("test", 0) == 0:
        hard_fail = True
        reasons.append("missing_val_or_test")
    if domain["odoo"] < 1 or domain["generic"] < 1:
        hard_fail = True
        reasons.append("domain_imbalance")
    if normalized["normalized_duplicate_groups"] > 5:
        hard_fail = True
        reasons.append("normalized_duplicates_severe")
    if not neg_ok:
        hard_fail = True
        reasons.append("negatives_mismatched")

    verdict = needs_fix_verdict if hard_fail else ready_verdict

    scorecard = {
        "native_records": len(records),
        "odoo": domain["odoo"],
        "generic": domain["generic"],
        "scenario_families": len(families),
        "largest_family_concentration": max(families.values()) if families else 0,
        "duplicate_groups": exact["duplicate_groups"],
        "normalized_duplicate_groups": normalized["normalized_duplicate_groups"],
        "forbidden_how": how_hits,
        "taxonomy_violations": tax_hits,
        "negative_contamination": neg_contam,
        "legacy_contamination": legacy_hits,
        "train": split_counts.get("train", 0),
        "validation": split_counts.get("validation", 0),
        "test": split_counts.get("test", 0),
        "family_leakage": len(leakage),
        "pack_validity": pack_ok,
        "provider_dataset_equivalence": provider_dataset_equiv,
        "by_record_type": dict(by_type),
        "decision_kinds": dict(decision_kinds),
        "negatives_ok": neg_ok,
    }

    return {
        "version": REASONING_SPARSE_VERSION,
        "provider_capability": provider,
        "verdict": verdict,
        "fail_reasons": reasons,
        "scorecard": scorecard,
        "by_record_type": dict(by_type),
        "by_family": dict(families),
        "domain": domain,
        "duplicates": exact,
        "normalized_duplicates": normalized,
        "splits": dict(split_counts),
        "split_rows": split_rows,
        "leakage": leakage,
        "split_strategy": document_split_strategy(),
        "pack_count": len(pack),
        "pack_examples": pack,
        "decision_kinds": dict(decision_kinds),
    }


def analyze_conversation_controlled(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _analyze_capability(
        records,
        provider="conversation",
        target_min=CONVERSATION_TARGET_MIN,
        target_max=CONVERSATION_TARGET_MAX,
        min_families=CONVERSATION_MIN_FAMILIES,
        ready_verdict="CONVERSATION_CORPUS_READY",
        needs_fix_verdict="CONVERSATION_CORPUS_NEEDS_FIXES",
    )


def analyze_approval_controlled(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _analyze_capability(
        records,
        provider="approval",
        target_min=APPROVAL_TARGET_MIN,
        target_max=APPROVAL_TARGET_MAX,
        min_families=APPROVAL_MIN_FAMILIES,
        ready_verdict="APPROVAL_CORPUS_READY",
        needs_fix_verdict="APPROVAL_CORPUS_NEEDS_FIXES",
    )


def _batch2_checksum() -> str:
    return json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))["checksum"]


def _write_capability_tree(
    *,
    root: Path,
    provider: str,
    generator: str,
    records: list[dict[str, Any]],
    analysis: dict[str, Any],
    pack: list[Any],
    split_rows: list[dict[str, Any]],
    family_count: int,
) -> dict[str, str]:
    written: dict[str, str] = {}
    root.mkdir(parents=True, exist_ok=True)
    resolved = root.resolve()
    assert "controlled_batch_2" not in str(resolved)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_type[str(r["record_type"])].append(r)

    native_name = f"{provider}_native.jsonl"
    write_jsonl(root / native_name, records)
    written[str(root / native_name)] = f"{len(records)} records"
    for rtype, rows in sorted(by_type.items()):
        write_jsonl(root / f"{rtype}.jsonl", rows)
        written[str(root / f"{rtype}.jsonl")] = f"{len(rows)} records"

    pack_rows = [
        {
            "example_id": ex.example_id,
            "dataset_type": ex.dataset_type.value,
            "messages": [dict(m) for m in ex.messages],
            "metadata": dict(ex.metadata),
        }
        for ex in pack
    ]
    write_jsonl(root / "pack_reasoning.jsonl", pack_rows)
    written[str(root / "pack_reasoning.jsonl")] = f"{len(pack_rows)} examples"
    write_jsonl(root / "splits.jsonl", split_rows)
    written[str(root / "splits.jsonl")] = f"{len(split_rows)} rows"

    scorecard = analysis["scorecard"]
    manifest = {
        "version": REASONING_SPARSE_VERSION,
        "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
        "provider_capability": provider,
        "product_plane": "reasoning",
        "generator": generator,
        "total_records": len(records),
        "by_record_type": {k: len(v) for k, v in sorted(by_type.items())},
        "scorecard": scorecard,
        "verdict": analysis["verdict"],
        "legacy_projection": False,
        "controlled_batch_2_modified": False,
        "split_version": "fp2-split-1.0.0",
        "notes": [
            f"AT-7.3 controlled {provider} corpus",
            "Independent from controlled_batch_2 and Planner",
            "Not a production adapter pack / not for certification",
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written[str(root / "manifest.json")] = "manifest"

    report = {k: v for k, v in analysis.items() if k not in {"pack_examples", "split_rows"}}
    (root / "quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    written[str(root / "quality_report.json")] = "quality_report"

    gen_meta = {
        "version": REASONING_SPARSE_VERSION,
        "provider_capability": provider,
        "family_count": family_count,
        "actual_count": len(records),
        "scenario_family_mechanism": (
            "metadata.scenario_family groups related variants; "
            "assign_split uses family: key (fp2-split-1.0.0)"
        ),
        "record_types": sorted(by_type.keys()),
    }
    (root / "generation_metadata.json").write_text(
        json.dumps(gen_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written[str(root / "generation_metadata.json")] = "generation_metadata"
    return written


def _write_evaluation_semantics_tree(root: Path) -> dict[str, str]:
    written: dict[str, str] = {}
    root.mkdir(parents=True, exist_ok=True)
    assert "controlled_batch_2" not in str(root.resolve())

    semantics = dict(EVALUATION_SEMANTIC_DEFINITION)
    (root / "semantics_report.json").write_text(
        json.dumps(semantics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written[str(root / "semantics_report.json")] = "semantics_report"

    manifest = {
        "version": REASONING_SPARSE_VERSION,
        "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
        "provider_capability": "evaluation",
        "product_plane": "reasoning",
        "generator": "evaluation_semantics",
        "total_records": 0,
        "verdict": EVALUATION_SEMANTIC_DEFINITION["status"],
        "fp2_generation_authorized": False,
        "record_type_authorized": EVALUATION_SEMANTIC_DEFINITION.get("record_type"),
        "legacy_projection": False,
        "controlled_batch_2_modified": False,
        "notes": [
            "AT-7.4 Evaluation mapping ready — evaluation_judgment only",
            "No native FP2 Evaluation corpus yet (await AT-7.5)",
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written[str(root / "manifest.json")] = "manifest"
    return written


def emit_reasoning_sparse_corpora(
    *,
    training_root: Path | None = None,
    datasets_root: Path | None = None,
) -> dict[str, Any]:
    """Write Conversation + Approval corpora and Evaluation semantics tree."""
    before = _batch2_checksum()
    assert before == BATCH2_CHECKSUM

    conv_records = generate_conversation_controlled_records()
    conv_analysis = analyze_conversation_controlled(conv_records)
    conv_pack = conv_analysis.pop("pack_examples")
    conv_splits = conv_analysis.pop("split_rows")

    appr_records = generate_approval_controlled_records()
    appr_analysis = analyze_approval_controlled(appr_records)
    appr_pack = appr_analysis.pop("pack_examples")
    appr_splits = appr_analysis.pop("split_rows")

    train_out = Path(training_root or _DEFAULT_TRAINING)
    data_out = Path(datasets_root or _DEFAULT_DATASETS)
    written: dict[str, str] = {}

    for base in (train_out, data_out):
        written.update(
            _write_capability_tree(
                root=base / "conversation",
                provider="conversation",
                generator=CONVERSATION_CONTROLLED_GENERATOR,
                records=conv_records,
                analysis=conv_analysis,
                pack=conv_pack,
                split_rows=conv_splits,
                family_count=conversation_family_count(),
            )
        )
        written.update(
            _write_capability_tree(
                root=base / "approval",
                provider="approval",
                generator=APPROVAL_CONTROLLED_GENERATOR,
                records=appr_records,
                analysis=appr_analysis,
                pack=appr_pack,
                split_rows=appr_splits,
                family_count=approval_family_count(),
            )
        )
        written.update(_write_evaluation_semantics_tree(base / "evaluation"))

    after = _batch2_checksum()
    assert after == before == BATCH2_CHECKSUM

    overall = "REASONING_SPARSE_DATA_PARTIAL"
    return {
        "version": REASONING_SPARSE_VERSION,
        "overall_verdict": overall,
        "conversation": {
            "count": len(conv_records),
            "verdict": conv_analysis["verdict"],
            "scorecard": conv_analysis["scorecard"],
            "fail_reasons": conv_analysis["fail_reasons"],
            "splits": conv_analysis["splits"],
        },
        "approval": {
            "count": len(appr_records),
            "verdict": appr_analysis["verdict"],
            "scorecard": appr_analysis["scorecard"],
            "fail_reasons": appr_analysis["fail_reasons"],
            "splits": appr_analysis["splits"],
        },
        "evaluation": {
            "count": 0,
            "verdict": EVALUATION_SEMANTIC_DEFINITION["status"],
            "status": EVALUATION_SEMANTIC_DEFINITION["status"],
        },
        "batch2_checksum_before": before,
        "batch2_checksum_after": after,
        "batch2_immutable": True,
        "written": written,
    }
