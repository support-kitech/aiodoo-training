"""Filesystem helpers for artifact output — create directories only when writing."""

from __future__ import annotations

from pathlib import Path


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory of ``path`` immediately before a write."""
    parent = path.parent
    if parent != Path(".") and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


__all__ = ["ensure_parent_dir"]
