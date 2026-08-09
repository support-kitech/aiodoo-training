#!/usr/bin/env python3
"""AT-6.3 — prepare deterministic Context smoke subset (does not modify corpus)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]
CORPUS = WORKSPACE / "aiodoo-datasets/datasets/fp2/context_controlled_1"
OUT = WORKSPACE / "aiodoo-training/artifacts/at6_context/smoke"


def corpus_checksum(root: Path) -> str:
    h = hashlib.sha256()
    for name in sorted(
        [
            "context_native.jsonl",
            "capability_intent.jsonl",
            "observation.jsonl",
            "pack_context.jsonl",
            "splits.jsonl",
            "manifest.json",
            "quality_report.json",
            "generation_metadata.json",
        ]
    ):
        p = root / name
        if p.is_file():
            h.update(name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def sort_key(rid: str) -> str:
    return hashlib.sha256(rid.encode("utf-8")).hexdigest()


def select(pool_meta: dict[str, dict], splits: dict[str, str], split_name: str, n: int) -> list[str]:
    pool = [rid for rid, sp in splits.items() if sp == split_name]
    by_type: dict[str, list[str]] = defaultdict(list)
    for rid in pool:
        by_type[pool_meta[rid]["record_type"]].append(rid)
    for t in by_type:
        by_type[t].sort(key=sort_key)
    types = sorted(by_type.keys())
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
    for rtype, q in quota.items():
        buckets: dict[str, list[str]] = defaultdict(list)
        for rid in by_type[rtype]:
            buckets[pool_meta[rid]["capability_id"] or "<none>"].append(rid)
        caps = sorted(buckets.keys(), key=lambda c: (sort_key(c), c))
        for c in caps:
            buckets[c].sort(key=sort_key)
        picked: list[str] = []
        idx = 0
        while len(picked) < q and any(buckets[c] for c in caps):
            cap = caps[idx % len(caps)]
            idx += 1
            if buckets[cap]:
                picked.append(buckets[cap].pop(0))
        selected.extend(picked)
    if len(selected) < n:
        remaining = sorted(set(pool) - set(selected), key=sort_key)
        selected.extend(remaining[: n - len(selected)])
    return selected[:n]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checksum = corpus_checksum(CORPUS)
    (OUT / "checksum_before.txt").write_text(checksum + "\n", encoding="utf-8")

    native: dict[str, dict] = {}
    for line in (CORPUS / "context_native.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        rid = rec["record_id"]
        if rec["record_type"] == "capability_intent":
            cap = rec["expected_output"]["capability_id"]
        else:
            cap = (rec.get("evidence") or {}).get("capability_id") or ""
        native[rid] = {
            "record_id": rid,
            "record_type": rec["record_type"],
            "provider_capability": rec["provider_capability"],
            "capability_id": cap,
            "domain_specialization": rec.get("domain_specialization"),
            "scenario_family": (rec.get("metadata") or {}).get("scenario_family"),
        }

    splits = {
        json.loads(line)["record_id"]: json.loads(line)["split"]
        for line in (CORPUS / "splits.jsonl").read_text(encoding="utf-8").splitlines()
    }
    # re-parse cleanly
    splits = {}
    for line in (CORPUS / "splits.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        splits[row["record_id"]] = row["split"]

    pack_by_id = {}
    for line in (CORPUS / "pack_context.jsonl").read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        assert ex.get("dataset_type") == "context"
        assert (ex.get("metadata") or {}).get("provider_capability") == "context"
        pack_by_id[(ex.get("metadata") or {})["record_id"]] = ex

    train_ids = select(native, splits, "train", 16)
    val_ids = select(native, splits, "validation", 4)

    def write_pack(path: Path, ids: list[str]) -> None:
        lines = [json.dumps(pack_by_id[i], sort_keys=True, ensure_ascii=False) for i in ids]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_pack(OUT / "context_train_smoke.jsonl", train_ids)
    write_pack(OUT / "context_validation_smoke.jsonl", val_ids)

    def dist(ids: list[str]) -> dict:
        return {
            "record_types": dict(Counter(native[i]["record_type"] for i in ids)),
            "capabilities": dict(Counter(native[i]["capability_id"] for i in ids)),
            "domains": dict(
                Counter((native[i]["domain_specialization"] or "<none>") for i in ids)
            ),
            "entries": [
                {
                    "record_id": i,
                    "record_type": native[i]["record_type"],
                    "provider_capability": native[i]["provider_capability"],
                    "capability_id": native[i]["capability_id"],
                    "domain_specialization": native[i]["domain_specialization"],
                    "scenario_family": native[i]["scenario_family"],
                    "source_split": splits[i],
                }
                for i in ids
            ],
        }

    manifest = {
        "phase": "AT-6.3",
        "adapter": "context",
        "provider_capability": "context",
        "product_plane": "development",
        "corpus_version": "fp2-context-controlled-1.0.0",
        "corpus_checksum": checksum,
        "source_pack": "pack_context",
        "source_train": 198,
        "source_validation": 52,
        "source_test": 11,
        "source_total": 261,
        "smoke_train": 16,
        "smoke_validation": 4,
        "selection_method": (
            "deterministic quotas by record_type then round-robin capability "
            "buckets ordered by sha256(record_id); strict subset of Context splits"
        ),
        "train_record_ids": train_ids,
        "validation_record_ids": val_ids,
        "train": dist(train_ids),
        "validation": dist(val_ids),
        "max_steps": 4,
        "filter_rule": "metadata.provider_capability==context AND dataset_type==context",
    }
    (OUT / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checksum": checksum, "train": len(train_ids), "val": len(val_ids)}, indent=2))


if __name__ == "__main__":
    main()
