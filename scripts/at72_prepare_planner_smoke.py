#!/usr/bin/env python3
"""AT-7.2 — prepare deterministic Planner smoke subset (does not modify corpus)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]
BATCH2 = WORKSPACE / "aiodoo-datasets/datasets/fp2/controlled_batch_2"
OUT = WORKSPACE / "aiodoo-training/artifacts/at7_planner/smoke"
BATCH2_CHECKSUM = "728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227"
CORPUS_VERSION = "fp2-controlled-2.0.0-tr7"


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


def eng_key(rec: dict[str, Any]) -> str:
    eo = rec.get("expected_output") if isinstance(rec.get("expected_output"), dict) else {}
    ev = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else {}
    cid = eo.get("capability_id") or ev.get("capability_id") or eo.get("decision_kind")
    if not cid:
        cid = eo.get("recommended_continuation") or ev.get("recommended_continuation")
    if not cid and isinstance(eo.get("steps"), list) and eo["steps"]:
        st = eo["steps"][0]
        if isinstance(st, dict):
            cid = st.get("action") or st.get("capability_id")
    return str(cid or "<none>")


def select(
    pool: list[tuple[str, dict[str, Any], dict[str, Any]]],
    n: int,
    *,
    avoid_fps: set[str] | None = None,
) -> list[str]:
    """Quotas by record_type, round-robin eng buckets, prefer unique fingerprints."""
    avoid_fps = set(avoid_fps or ())
    by_type: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for item in pool:
        by_type[item[1]["record_type"]].append(item)
    for t in by_type:
        by_type[t].sort(key=lambda x: sort_key(x[0]))

    types = sorted(by_type.keys())
    if not types:
        return []
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

    selected: list[str] = []
    used_fps: set[str] = set()
    for rtype, q in quota.items():
        buckets: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for item in by_type[rtype]:
            buckets[eng_key(item[1])].append(item)
        caps = sorted(buckets.keys(), key=lambda c: (sort_key(c), c))
        for c in caps:
            buckets[c].sort(key=lambda x: sort_key(x[0]))
        picked: list[str] = []
        idx = 0
        # First pass: skip fingerprints already used / avoided
        while len(picked) < q and any(buckets[c] for c in caps):
            cap = caps[idx % len(caps)]
            idx += 1
            if not buckets[cap]:
                continue
            rid, rec, _ex = buckets[cap].pop(0)
            fp = fingerprint(rec)
            if fp in used_fps or fp in avoid_fps:
                continue
            picked.append(rid)
            used_fps.add(fp)
        # Fill remaining if quota unmet (allow previously skipped only if needed)
        if len(picked) < q:
            leftovers = sorted(by_type[rtype], key=lambda x: sort_key(x[0]))
            for rid, rec, _ex in leftovers:
                if rid in picked:
                    continue
                fp = fingerprint(rec)
                if fp in used_fps:
                    continue
                picked.append(rid)
                used_fps.add(fp)
                if len(picked) >= q:
                    break
        selected.extend(picked)

    if len(selected) < n:
        remaining = [
            rid
            for rid, rec, _ in sorted(pool, key=lambda x: sort_key(x[0]))
            if rid not in selected and fingerprint(rec) not in used_fps
        ]
        selected.extend(remaining[: n - len(selected)])
    return selected[:n]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(parents=True, exist_ok=True)

    man = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))
    checksum = str(man.get("checksum") or "")
    assert checksum == BATCH2_CHECKSUM
    (OUT / "checksum_before.txt").write_text(checksum + "\n", encoding="utf-8")

    # Native planner index
    natives: dict[str, dict[str, Any]] = {}
    for name in (
        "capability_intent.jsonl",
        "execution_work_unit.jsonl",
        "planning_decision.jsonl",
        "observation.jsonl",
        "engineering_feedback.jsonl",
        "engineering_state.jsonl",
        "decision_context.jsonl",
        "loop_decision.jsonl",
    ):
        for line in (BATCH2 / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("provider_capability") != "planner":
                continue
            natives[str(rec["record_id"])] = rec

    splits: dict[str, str] = {}
    for line in (BATCH2 / "splits.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        splits[str(row["record_id"])] = str(row["split"])

    # Reasoning pack planner examples only (A≡B)
    pack_by_id: dict[str, dict[str, Any]] = {}
    for line in (BATCH2 / "pack_reasoning.jsonl").read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        meta = ex.get("metadata") or {}
        if meta.get("provider_capability") != "planner":
            continue
        if ex.get("dataset_type") != "planner":
            continue
        rid = str(meta.get("record_id") or "")
        assert rid in natives, rid
        assert natives[rid].get("provider_capability") == "planner"
        pack_by_id[rid] = ex

    assert len(pack_by_id) == 580

    # Duplicate groups among natives (AT-7.1 observation)
    buckets: dict[str, list[str]] = defaultdict(list)
    for rid, rec in natives.items():
        buckets[fingerprint(rec)].append(rid)
    dup_groups = {fp: ids for fp, ids in buckets.items() if len(ids) > 1}

    def pool_for(split_name: str) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for rid, ex in pack_by_id.items():
            if splits.get(rid) != split_name:
                continue
            # Strict subset of authoritative planner split population
            assert natives[rid].get("provider_capability") == "planner"
            out.append((rid, natives[rid], ex))
        return out

    train_pool = pool_for("train")
    val_pool = pool_for("validation")
    assert len(train_pool) == 460
    assert len(val_pool) == 59

    train_ids = select(train_pool, 16)
    val_ids = select(val_pool, 4, avoid_fps={fingerprint(natives[i]) for i in train_ids})
    assert len(train_ids) == 16 and len(val_ids) == 4
    assert set(train_ids).isdisjoint(val_ids)
    assert all(splits[i] == "train" for i in train_ids)
    assert all(splits[i] == "validation" for i in val_ids)

    def write_pack(path: Path, ids: list[str]) -> None:
        lines = [json.dumps(pack_by_id[i], sort_keys=True, ensure_ascii=False) for i in ids]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_pack(OUT / "data" / "planner_train_smoke.jsonl", train_ids)
    write_pack(OUT / "data" / "planner_validation_smoke.jsonl", val_ids)

    def dist(ids: list[str]) -> dict[str, Any]:
        entries = []
        for rid in ids:
            rec = natives[rid]
            meta = rec.get("metadata") or {}
            entries.append(
                {
                    "record_id": rid,
                    "record_type": rec.get("record_type"),
                    "provider_capability": rec.get("provider_capability"),
                    "dataset_type": "planner",
                    "engineering_or_decision": eng_key(rec),
                    "domain_specialization": rec.get("domain_specialization"),
                    "scenario_family": meta.get("scenario_family") or meta.get("scenario"),
                    "source_split": splits[rid],
                    "fingerprint": fingerprint(rec),
                }
            )
        return {
            "record_types": dict(Counter(e["record_type"] for e in entries)),
            "engineering_or_decision": dict(
                Counter(e["engineering_or_decision"] for e in entries)
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
    smoke_fps = {e["fingerprint"] for e in train_dist["entries"] + val_dist["entries"]}
    dup_in_smoke = {
        fp: dup_groups[fp]
        for fp in smoke_fps
        if fp in dup_groups
    }

    manifest = {
        "phase": "AT-7.2",
        "adapter": "planner",
        "provider_capability": "planner",
        "product_plane": "reasoning",
        "corpus_version": CORPUS_VERSION,
        "corpus_checksum": checksum,
        "source_pack": "pack_reasoning",
        "split_version": "fp2-split-1.0.0",
        "filter_rule": "metadata.provider_capability==planner AND dataset_type==planner",
        "native_planner_count": len(natives),
        "reasoning_pack_planner_count": len(pack_by_id),
        "authoritative_train_count": 524,
        "authoritative_validation_count": 62,
        "authoritative_test_count": 71,
        "reasoning_pack_train_count": len(train_pool),
        "reasoning_pack_validation_count": len(val_pool),
        "smoke_train": 16,
        "smoke_validation": 4,
        "selection_method": (
            "deterministic quotas by record_type then round-robin "
            "engineering/decision buckets ordered by sha256(record_id); "
            "prefer unique fingerprints; strict subset of Reasoning-pack "
            "Planner examples whose record_id is in authoritative train/val splits"
        ),
        "train_record_ids": train_ids,
        "validation_record_ids": val_ids,
        "train": train_dist,
        "validation": val_dist,
        "duplicate_groups_in_corpus": len(dup_groups),
        "duplicate_groups_present_in_smoke": {
            fp: ids for fp, ids in sorted(dup_in_smoke.items())
        },
        "duplicate_group_count_in_smoke": len(dup_in_smoke),
        "foundation_model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "max_steps": 4,
    }
    (OUT / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "checksum": checksum,
                "reasoning_pack_planner": len(pack_by_id),
                "train": len(train_ids),
                "val": len(val_ids),
                "train_types": train_dist["record_types"],
                "dup_in_smoke": len(dup_in_smoke),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
