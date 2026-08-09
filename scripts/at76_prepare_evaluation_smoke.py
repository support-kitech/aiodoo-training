#!/usr/bin/env python3
"""AT-7.6 — prepare deterministic Evaluation smoke subset (does not modify corpus)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]
CORPUS = WORKSPACE / "aiodoo-datasets/datasets/fp2/reasoning_controlled_1/evaluation"
OUT = WORKSPACE / "aiodoo-training/artifacts/at7_evaluation/smoke"
CORPUS_VERSION = "fp2-evaluation-controlled-1.0.0"
CORPUS_CHECKSUM = "764dba2849519c2b3cf1f5ff24acb84c644f3506b99dbc958762e470310e0883"
BATCH2_CHECKSUM = "728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227"
BATCH2 = WORKSPACE / "aiodoo-datasets/datasets/fp2/controlled_batch_2"


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
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def field_pattern(rec: dict[str, Any]) -> str:
    inp = rec.get("input") if isinstance(rec.get("input"), dict) else {}
    has_e = "expectation" in inp
    has_r = "rubric" in inp
    if has_e and has_r:
        return "candidate_expectation_rubric"
    if has_e:
        return "candidate_expectation"
    if has_r:
        return "candidate_rubric"
    return "candidate_only"


def select(
    pool: list[tuple[str, dict[str, Any], dict[str, Any]]],
    n: int,
    *,
    avoid_families: set[str] | None = None,
    avoid_fps: set[str] | None = None,
) -> list[str]:
    """Deterministic quotas by verdict, then round-robin candidate_category.

    Prefers at most one record per scenario_family to maximize family diversity.
    """
    avoid_families = set(avoid_families or ())
    avoid_fps = set(avoid_fps or ())

    by_verdict: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for item in pool:
        fam = str((item[1].get("metadata") or {}).get("scenario_family") or "")
        if fam in avoid_families:
            continue
        verdict = str((item[1].get("expected_output") or {}).get("verdict") or "")
        by_verdict[verdict].append(item)
    for v in by_verdict:
        by_verdict[v].sort(key=lambda x: sort_key(x[0]))

    verdicts = [v for v in ("pass", "fail", "inconclusive") if v in by_verdict]
    if not verdicts:
        return []
    quota = {v: n // len(verdicts) for v in verdicts}
    for v in verdicts:
        if sum(quota.values()) < n:
            quota[v] += 1
        if sum(quota.values()) >= n:
            break
    while sum(quota.values()) > n:
        for v in sorted(quota, key=lambda x: -quota[x]):
            if quota[v] > 0:
                quota[v] -= 1
                break

    selected: list[str] = []
    used_fps: set[str] = set()
    used_families: set[str] = set()

    for verdict, q in quota.items():
        buckets: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for item in by_verdict[verdict]:
            cat = str((item[1].get("metadata") or {}).get("candidate_category") or "<none>")
            buckets[cat].append(item)
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
            picked.append(rid)
            used_fps.add(fp)
            used_families.add(fam)
        # Fill remaining for this verdict if needed (still unique family/fp)
        if len(picked) < q:
            leftovers = sorted(by_verdict[verdict], key=lambda x: sort_key(x[0]))
            for rid, rec, _ex in leftovers:
                if rid in picked:
                    continue
                fam = str((rec.get("metadata") or {}).get("scenario_family") or "")
                fp = fingerprint(rec)
                if fp in used_fps or fam in used_families:
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
            if fam in avoid_families or fam in used_families or fp in used_fps:
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
    for line in (CORPUS / "evaluation_native.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        assert rec.get("provider_capability") == "evaluation"
        assert rec.get("record_type") == "evaluation_judgment"
        natives[str(rec["record_id"])] = rec

    splits: dict[str, str] = {}
    for line in (CORPUS / "splits.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        splits[str(row["record_id"])] = str(row["split"])

    pack_by_id: dict[str, dict[str, Any]] = {}
    for line in (CORPUS / "pack_evaluation.jsonl").read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        meta = ex.get("metadata") or {}
        assert ex.get("dataset_type") == "evaluation"
        assert meta.get("provider_capability") == "evaluation"
        assert meta.get("record_type") == "evaluation_judgment"
        rid = str(meta.get("record_id") or "")
        assert rid in natives
        pack_by_id[rid] = ex

    assert len(pack_by_id) == 252 == len(natives)

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
    assert len(train_pool) == 200
    assert len(val_pool) == 48
    assert len(test_pool) == 4

    test_families = {
        str((rec.get("metadata") or {}).get("scenario_family") or "")
        for _, rec, _ in test_pool
    }
    train_ids = select(train_pool, 16, avoid_families=test_families)
    train_families = {
        str((natives[i].get("metadata") or {}).get("scenario_family") or "") for i in train_ids
    }
    val_ids = select(
        val_pool,
        4,
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

    def write_pack(path: Path, ids: list[str]) -> None:
        lines = [json.dumps(pack_by_id[i], sort_keys=True, ensure_ascii=False) for i in ids]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_pack(OUT / "data" / "evaluation_train_smoke.jsonl", train_ids)
    write_pack(OUT / "data" / "evaluation_validation_smoke.jsonl", val_ids)

    def dist(ids: list[str]) -> dict[str, Any]:
        entries = []
        for rid in ids:
            rec = natives[rid]
            meta = rec.get("metadata") or {}
            out = rec.get("expected_output") or {}
            inp = rec.get("input") or {}
            entries.append(
                {
                    "record_id": rid,
                    "record_type": rec.get("record_type"),
                    "provider_capability": rec.get("provider_capability"),
                    "dataset_type": "evaluation",
                    "verdict": out.get("verdict"),
                    "score": out.get("score"),
                    "has_expectation": "expectation" in inp,
                    "has_rubric": "rubric" in inp,
                    "has_explanation": "explanation" in out,
                    "field_pattern": field_pattern(rec),
                    "candidate_category": meta.get("candidate_category"),
                    "domain_specialization": rec.get("domain_specialization"),
                    "scenario_family": meta.get("scenario_family"),
                    "source_split": splits[rid],
                    "fingerprint": fingerprint(rec),
                }
            )
        return {
            "verdicts": dict(Counter(e["verdict"] for e in entries)),
            "field_patterns": dict(Counter(e["field_pattern"] for e in entries)),
            "candidate_categories": dict(Counter(e["candidate_category"] for e in entries)),
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

    manifest = {
        "phase": "AT-7.6",
        "adapter": "evaluation",
        "provider_capability": "evaluation",
        "product_plane": "reasoning",
        "record_type": "evaluation_judgment",
        "corpus_version": CORPUS_VERSION,
        "corpus_checksum": checksum,
        "source_pack": "pack_evaluation",
        "split_version": "fp2-split-1.0.0",
        "filter_rule": (
            "metadata.provider_capability==evaluation AND dataset_type==evaluation "
            "AND record_type==evaluation_judgment"
        ),
        "native_count": 252,
        "authoritative_train_count": 200,
        "authoritative_validation_count": 48,
        "authoritative_test_count": 4,
        "smoke_train": 16,
        "smoke_validation": 4,
        "selection_method": (
            "deterministic quotas by verdict then round-robin candidate_category "
            "ordered by sha256(record_id); prefer unique fingerprints and at most "
            "one record per scenario_family; strict subset of train/validation; "
            "disjoint from corpus test families"
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
                "train_verdicts": train_dist["verdicts"],
                "train_categories": train_dist["candidate_categories"],
                "family_isolation_ok": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
