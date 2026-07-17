#!/usr/bin/env python3
"""Cleanup utility for AIODOO Drive workspace artifact directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aiodoo_training.artifacts.cleanup import cleanup_workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove empty, stale, and abandoned artifact directories from the Drive workspace."
        ),
    )
    parser.add_argument(
        "workspace",
        type=Path,
        nargs="?",
        default=None,
        help="AIODOO workspace root (default: AIODOO_WORKSPACE_ROOT env var).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Perform deletions (default is dry-run).",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="Number of latest checkpoints to retain per experiment (default: 3).",
    )
    args = parser.parse_args(argv)

    root = args.workspace
    if root is None:
        import os

        raw = os.environ.get("AIODOO_WORKSPACE_ROOT")
        if not raw:
            print("Error: provide workspace path or set AIODOO_WORKSPACE_ROOT.", file=sys.stderr)
            return 2
        root = Path(raw)

    dry_run = not args.delete
    report = cleanup_workspace(
        root,
        dry_run=dry_run,
        delete=args.delete,
        keep_checkpoints=args.keep_checkpoints,
    )

    mode = "DRY-RUN" if dry_run else "DELETE"
    print(f"Cleanup ({mode}) workspace={report.workspace_root}")
    print(f"  empty_dirs: {len(report.empty_dirs)}")
    print(f"  stale_tmp_dirs: {len(report.stale_tmp_dirs)}")
    print(f"  abandoned_checkpoints: {len(report.abandoned_checkpoints)}")
    if report.deleted:
        print(f"  deleted: {len(report.deleted)}")

    for label, paths in (
        ("empty", report.empty_dirs),
        ("stale_tmp", report.stale_tmp_dirs),
        ("abandoned_checkpoint", report.abandoned_checkpoints),
        ("deleted", report.deleted),
    ):
        for path in paths:
            print(f"  [{label}] {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
