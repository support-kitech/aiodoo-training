#!/usr/bin/env python3
"""AT-7.7 — prepare deterministic Conversation smoke subset (does not modify corpus)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]
CORPUS = WORKSPACE / "aiodoo-datasets/datasets/fp2/reasoning_controlled_1/conversation"
OUT = WORKSPACE / "aiodoo-training/artifacts/at7_conversation/smoke"
CORPUS_VERSION = "fp2-reasoning-sparse-1.0.0"
CORPUS_CHECKSUM = "488b3a7576071c875c32e277c49562bb9c472904e32b12a1b98fcf6558da9de3"
BATCH2_CHECKSUM = "728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227"
BATCH2 = WORKSPACE / "aiodoo-datasets/datasets/fp2/controlled_batch_2"
AUTHORIZED_TYPES = frozenset({"decision_context", "loop_decision"})


def corpus_tree_checksum(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix().encode()
            h.update(rel)
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def sort_key(rid: str) -> str:
    return hashlib.sha256(rid.encode("utf-8")).hexdigest()


def fingerprint(rec: dict[str, Any]) -> str:
    payload = {
        "record_type": rec.get("record_type"),
        "provider_capability": rec.get("provider_capability"),
        "domain_specialization": rec.get("domain_specialization"),
        "input": rec.get("input"),
        "expected_output": rec.get("expected_output"),
        "evidence": rec.get("evidence"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def select(
    pool: list[tuple[str, dict[str, Any], dict[str, Any]]],
    n: int,
    *,
    type_quota: dict[str, int] | None = None,
    avoid_families: set[str] | None = None,
    avoid_fps: set[str] | None = None,
) -> list[str]:
    """Deterministic quotas by record_type, then round-robin domain.

    Prefers at most one record per scenario_family.
    """
    avoid_families = set(avoid_families or ())
    avoid_fps = set(avoid_fps or ())

    by_type: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for item in pool:
        fam = str((item[1].get("metadata") or {}).get("scenario_family") or "")
        if fam in avoid_families:
            continue
        rtype = str(item[1].get("record_type") or "")
        if rtype not in AUTHORIZED_TYPES:
            continue
        by_type[rtype].append(item)
    for t in by_type:
        by_type[t].sort(key=lambda x: sort_key(x[0]))

    types = sorted(by_type.keys())
    if not types:
        return []

    if type_quota is None:
        quota = {t: n // len(types) for t in types}
        for t in types:
            if sum(quota.values()) < n:
                quota[t] += 1
            if sum(quota.values()) >= n:
                break
        while sum(quota.values()) > n:
            for t in sorted(quota, key=lambda x: -quota[x]):
                if quota[t] > 0:
                    quota[t] -= 1
                    break
    else:
        quota = {t: int(type_quota.get(t, 0)) for t in types}
        # Fill remaining slots if quota underspecified
        while sum(quota.values()) < n:
            for t in types:
                if sum(quota.values()) >= n:
                    break
                quota[t] += 1
        while sum(quota.values()) > n:
            for t in sorted(quota, key=lambda x: -quota[x]):
                if quota[t] > 0:
                    quota[t] -= 1
                    break

    selected: list[str] = []
    used_fps: set[str] = set()
    used_families: set[str] = set()

    for rtype, q in quota.items():
        if q <= 0 or rtype not in by_type:
            continue
        buckets: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for item in by_type[rtype]:
            domain = str(item[1].get("domain_specialization") or "generic")
            buckets[domain].append(item)
        caps = sorted(buckets.keys(), key=lambda c: (sort_key(c), c))
        for c in caps:
            buckets[c].sort(key=lambda x: sort_key(x[0]))
        picked: list[str] = []
        idx = 0
        while len(picked) < q and any(buckets[c] for c in caps):
            cap = caps[idx % len(caps)]
            idx += 1
            if not buckets[cap]:
                continue
            rid, rec, _ex = buckets[cap].pop(0)
            fam = str((rec.get("metadata") or {}).get("scenario_family") or "")
            fp = fingerprint(rec)
            if fp in used_fps or fp in avoid_fps or fam in used_families:
                continue
            # loop_decision must be clarify for Conversation corpus
            if rtype == "loop_decision":
                kind = str((rec.get("expected_output") or {}).get("decision_kind") or "")
                if kind != "clarify":
                    continue
            picked.append(rid)
            used_fps.add(fp)
            used_families.add(fam)
        if len(picked) < q:
            leftovers = sorted(by_type[rtype], key=lambda x: sort_key(x[0]))
            for rid, rec, _ex in leftovers:
                if rid in picked:
                    continue
                fam = str((rec.get("metadata") or {}).get("scenario_family") or "")
                fp = fingerprint(rec)
                if fp in used_fps or fam in used_families:
                    continue
                if rtype == "loop_decision":
                    kind = str((rec.get("expected_output") or {}).get("decision_kind") or "")
                    if kind != "clarify":
                        continue
                picked.append(rid)
                used_fps.add(fp)
                used_families.add(fam)
                if len(picked) >= q:
                    break
        selected.extend(picked)

    if len(selected) < n:
        remaining = sorted(pool, key=lambda x: sort_key(x[0]))
        for rid, rec, _ex in remaining:
            if rid in selected:
                continue
            fam = str((rec.get("metadata") or {}).get("scenario_family") or "")
            fp = fingerprint(rec)
            rtype = str(rec.get("record_type") or "")
            if fam in avoid_families or fam in used_families or fp in used_fps:
                continue
            if rtype not in AUTHORIZED_TYPES:
                continue
            if rtype == "loop_decision":
                kind = str((rec.get("expected_output") or {}).get("decision_kind") or "")
                if kind != "clarify":
                    continue
            selected.append(rid)
            used_fps.add(fp)
            used_families.add(fam)
            if len(selected) >= n:
                break
    return selected[:n]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(parents=True, exist_ok=True)

    checksum = corpus_tree_checksum(CORPUS)
    assert checksum == CORPUS_CHECKSUM, (checksum, CORPUS_CHECKSUM)
    (OUT / "checksum_before.txt").write_text(checksum + "\n", encoding="utf-8")

    batch2_man = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))
    assert batch2_man.get("checksum") == BATCH2_CHECKSUM

    natives: dict[str, dict[str, Any]] = {}
    for line in (CORPUS / "conversation_native.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        assert rec.get("provider_capability") == "conversation"
        assert rec.get("record_type") in AUTHORIZED_TYPES
        natives[str(rec["record_id"])] = rec

    splits: dict[str, str] = {}
    for line in (CORPUS / "splits.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        splits[str(row["record_id"])] = str(row["split"])

    pack_by_id: dict[str, dict[str, Any]] = {}
    for line in (CORPUS / "pack_reasoning.jsonl").read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        meta = ex.get("metadata") or {}
        assert ex.get("dataset_type") == "conversation"
        assert meta.get("provider_capability") == "conversation"
        rid = str(meta.get("record_id") or "")
        assert rid in natives
        pack_by_id[rid] = ex

    assert len(pack_by_id) == 232 == len(natives)

    def pool_for(split_name: str) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for rid, ex in pack_by_id.items():
            if splits.get(rid) != split_name:
                continue
            out.append((rid, natives[rid], ex))
        return out

    train_pool = pool_for("train")
    val_pool = pool_for("validation")
    test_pool = pool_for("test")
    assert len(train_pool) == 180
    assert len(val_pool) == 28
    assert len(test_pool) == 24

    test_families = {
        str((rec.get("metadata") or {}).get("scenario_family") or "")
        for _, rec, _ in test_pool
    }
    train_ids = select(
        train_pool,
        16,
        type_quota={"decision_context": 8, "loop_decision": 8},
        avoid_families=test_families,
    )
    train_families = {
        str((natives[i].get("metadata") or {}).get("scenario_family") or "") for i in train_ids
    }
    val_ids = select(
        val_pool,
        4,
        type_quota={"decision_context": 2, "loop_decision": 2},
        avoid_families=test_families | train_families,
        avoid_fps={fingerprint(natives[i]) for i in train_ids},
    )
    val_families = {
        str((natives[i].get("metadata") or {}).get("scenario_family") or "") for i in val_ids
    }

    assert len(train_ids) == 16 and len(val_ids) == 4
    assert set(train_ids).isdisjoint(val_ids)
    assert all(splits[i] == "train" for i in train_ids)
    assert all(splits[i] == "validation" for i in val_ids)
    assert train_families.isdisjoint(val_families)
    assert train_families.isdisjoint(test_families)
    assert val_families.isdisjoint(test_families)

    # Semantic guards
    for rid in train_ids + val_ids:
        rec = natives[rid]
        assert rec["provider_capability"] == "conversation"
        assert rec["record_type"] in AUTHORIZED_TYPES
        if rec["record_type"] == "loop_decision":
            assert rec["expected_output"]["decision_kind"] == "clarify"
            assert rec["expected_output"]["decision_kind"] not in {
                "approve",
                "reject",
                "modify",
            }
        eo = rec.get("expected_output") or {}
        assert "verdict" not in eo  # not Evaluation
        assert eo.get("decision_kind") != "replan" or rec["record_type"] != "loop_decision"

    def write_pack(path: Path, ids: list[str]) -> None:
        lines = [json.dumps(pack_by_id[i], sort_keys=True, ensure_ascii=False) for i in ids]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_pack(OUT / "data" / "conversation_train_smoke.jsonl", train_ids)
    write_pack(OUT / "data" / "conversation_validation_smoke.jsonl", val_ids)

    def dist(ids: list[str]) -> dict[str, Any]:
        entries = []
        for rid in ids:
            rec = natives[rid]
            meta = rec.get("metadata") or {}
            eo = rec.get("expected_output") or {}
            entries.append(
                {
                    "record_id": rid,
                    "record_type": rec.get("record_type"),
                    "provider_capability": rec.get("provider_capability"),
                    "dataset_type": "conversation",
                    "decision_kind": eo.get("decision_kind"),
                    "domain_specialization": rec.get("domain_specialization"),
                    "scenario_family": meta.get("scenario_family"),
                    "source_split": splits[rid],
                    "fingerprint": fingerprint(rec),
                }
            )
        return {
            "record_types": dict(Counter(e["record_type"] for e in entries)),
            "decision_kinds": dict(
                Counter((e["decision_kind"] or "<none>") for e in entries)
            ),
            "domains": dict(
                Counter((e["domain_specialization"] or "<none>") for e in entries)
            ),
            "scenario_families": dict(
                Counter((e["scenario_family"] or "<none>") for e in entries)
            ),
            "entries": entries,
        }

    train_dist = dist(train_ids)
    val_dist = dist(val_ids)
    smoke_fps = [e["fingerprint"] for e in train_dist["entries"] + val_dist["entries"]]
    assert len(smoke_fps) == len(set(smoke_fps))

    # Preferred 8/8 if achievable
    assert train_dist["record_types"].get("decision_context") == 8
    assert train_dist["record_types"].get("loop_decision") == 8
    assert train_dist["decision_kinds"].get("clarify") == 8

    manifest = {
        "phase": "AT-7.7",
        "adapter": "conversation",
        "provider_capability": "conversation",
        "product_plane": "reasoning",
        "authorized_record_types": sorted(AUTHORIZED_TYPES),
        "corpus_version": CORPUS_VERSION,
        "corpus_checksum": checksum,
        "source_pack": "pack_reasoning",
        "split_version": "fp2-split-1.0.0",
        "filter_rule": (
            "metadata.provider_capability==conversation AND dataset_type==conversation"
        ),
        "native_count": 232,
        "authoritative_train_count": 180,
        "authoritative_validation_count": 28,
        "authoritative_test_count": 24,
        "smoke_train": 16,
        "smoke_validation": 4,
        "selection_method": (
            "deterministic quotas by record_type (prefer 8 decision_context + "
            "8 loop_decision/clarify for train; 2+2 for val) then round-robin "
            "domain ordered by sha256(record_id); unique fingerprints; at most "
            "one record per scenario_family; disjoint from corpus test families"
        ),
        "train_record_ids": train_ids,
        "validation_record_ids": val_ids,
        "train": train_dist,
        "validation": val_dist,
        "family_isolation": {
            "train_families": sorted(train_families),
            "validation_families": sorted(val_families),
            "test_families": sorted(test_families),
            "train_val_intersection": [],
            "train_test_intersection": [],
            "val_test_intersection": [],
        },
        "duplicate_fingerprints_in_smoke": 0,
        "foundation_model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "max_steps": 4,
        "batch2_checksum": BATCH2_CHECKSUM,
        "legacy_projection": "NOT PERFORMED",
        "approval_contamination": 0,
        "planner_contamination": 0,
        "evaluation_contamination": 0,
    }
    (OUT / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "checksum": checksum,
                "train": len(train_ids),
                "val": len(val_ids),
                "train_types": train_dist["record_types"],
                "train_kinds": train_dist["decision_kinds"],
                "train_domains": train_dist["domains"],
                "family_isolation_ok": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
