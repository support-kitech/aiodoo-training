"""Portable per-rank seed derivation."""

from __future__ import annotations

import hashlib


def derive_rank_seed(root_seed: int, global_rank: int, *, epoch: int = 0) -> int:
    """Derive a portable per-rank seed from root_seed + rank (+ optional epoch)."""
    raw = f"{root_seed}|{global_rank}|{epoch}".encode()
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)
