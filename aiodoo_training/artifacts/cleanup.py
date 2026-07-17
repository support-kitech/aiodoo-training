"""Workspace cleanup utility — remove empty and stale artifact directories."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

_STALE_TMP_PREFIX = ".tmp-"
_ABANDONED_CHECKPOINT_PREFIX = "checkpoint-"
_BACKUP_PREFIX = ".backup-"

# Production trees that must never be deleted by cleanup.
_PROTECTED_RELATIVE_PREFIXES: tuple[str, ...] = (
    "models/adapters",
    "models/merged",
    "models/exports",
    "experiments",
)


@dataclass(frozen=True, slots=True)
class CleanupReport:
    """Summary of a cleanup scan or delete operation."""

    workspace_root: Path
    dry_run: bool
    empty_dirs: tuple[Path, ...] = ()
    stale_tmp_dirs: tuple[Path, ...] = ()
    abandoned_checkpoints: tuple[Path, ...] = ()
    deleted: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def total_candidates(self) -> int:
        return len(self.empty_dirs) + len(self.stale_tmp_dirs) + len(self.abandoned_checkpoints)


def is_protected_path(path: Path, workspace_root: Path) -> bool:
    """Return whether ``path`` is under a protected production artifact tree."""
    try:
        relative = path.resolve().relative_to(workspace_root.resolve())
    except ValueError:
        return True
    rel = relative.as_posix()
    for prefix in _PROTECTED_RELATIVE_PREFIXES:
        if rel == prefix or rel.startswith(f"{prefix}/"):
            return True
    return False


def scan_empty_directories(root: Path, *, workspace_root: Path | None = None) -> list[Path]:
    """
    Find empty directories under runtime cache only.

    Never scans or returns protected production trees (adapters, merged,
    exports, experiments).
    """
    if not root.is_dir():
        return []
    ws_root = workspace_root or root
    cache_root = root / "training" / "cache"
    if not cache_root.is_dir():
        return []

    empty: list[Path] = []
    for path in sorted(cache_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.is_dir():
            continue
        if is_protected_path(path, ws_root):
            continue
        if path.name.startswith(_STALE_TMP_PREFIX) or path.name.startswith(_BACKUP_PREFIX):
            continue
        if not any(path.iterdir()):
            empty.append(path)
    return empty


def scan_stale_tmp_dirs(root: Path, *, workspace_root: Path | None = None) -> list[Path]:
    """Find incomplete temporary publish/export/checkpoint directories."""
    if not root.is_dir():
        return []
    ws_root = workspace_root or root
    stale: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if is_protected_path(path, ws_root):
            continue
        name = path.name
        if (
            name.startswith(_STALE_TMP_PREFIX)
            or name.startswith(".tmp-publish-")
            or name.startswith(".tmp-merged-")
            or name.startswith(_BACKUP_PREFIX)
        ):
            stale.append(path)
    return sorted(stale)


def scan_abandoned_checkpoints(
    root: Path,
    *,
    keep_latest: int = 3,
) -> list[Path]:
    """
    Find checkpoint directories beyond retention under ``training/cache/``.

    Only removes checkpoints when more than ``keep_latest`` exist in the same
    parent directory. Never touches published adapter, merged, or export trees.
    """
    if not root.is_dir():
        return []
    abandoned: list[Path] = []
    cache_root = root / "training" / "cache"
    if not cache_root.is_dir():
        return []
    for exp_dir in cache_root.iterdir():
        if not exp_dir.is_dir():
            continue
        ckpt_dir = exp_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            continue
        checkpoints: list[tuple[int, Path]] = []
        for child in ckpt_dir.iterdir():
            if child.is_dir() and child.name.startswith(_ABANDONED_CHECKPOINT_PREFIX):
                try:
                    step = int(child.name.split("-", 1)[1])
                except (IndexError, ValueError):
                    continue
                checkpoints.append((step, child))
        checkpoints.sort(key=lambda item: item[0])
        if len(checkpoints) > keep_latest:
            for _, path in checkpoints[: len(checkpoints) - keep_latest]:
                abandoned.append(path)
    return abandoned


def cleanup_workspace(
    root: Path,
    *,
    dry_run: bool = True,
    delete: bool = False,
    keep_checkpoints: int = 3,
) -> CleanupReport:
    """
    Scan and optionally remove empty, stale, and abandoned artifact directories.

    Parameters
    ----------
    dry_run:
        When True (default), report candidates without deleting.
    delete:
        When True, perform deletions. Ignored when ``dry_run`` is True.
    """
    empty = tuple(scan_empty_directories(root, workspace_root=root))
    stale = tuple(scan_stale_tmp_dirs(root, workspace_root=root))
    abandoned = tuple(scan_abandoned_checkpoints(root, keep_latest=keep_checkpoints))

    deleted: list[Path] = []
    if not dry_run and delete:
        for path in (*stale, *abandoned, *empty):
            if is_protected_path(path, root):
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                deleted.append(path)
            elif path.is_file():
                path.unlink(missing_ok=True)
                deleted.append(path)

    return CleanupReport(
        workspace_root=root,
        dry_run=dry_run,
        empty_dirs=empty,
        stale_tmp_dirs=stale,
        abandoned_checkpoints=abandoned,
        deleted=tuple(deleted),
    )


__all__ = [
    "CleanupReport",
    "cleanup_workspace",
    "is_protected_path",
    "scan_abandoned_checkpoints",
    "scan_empty_directories",
    "scan_stale_tmp_dirs",
]
