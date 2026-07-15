"""Golden: identical export inputs → identical fingerprints and checksums."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase4
from aiodoo_training.export import build_stub_export_context, run_stub_export


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase4(overwrite=True)


_VOLATILE_REL_PATHS = frozenset({"export_manifest.json", "checksums.sha256"})


def _portable_manifest_fields(manifest) -> dict:
    """Manifest fields stable across output dirs (exclude timestamps/software)."""
    artifacts = sorted(
        (
            d.role,
            d.relative_path,
            d.checksum,
            d.required,
            d.content_type,
        )
        for d in manifest.artifacts
        if d.relative_path not in _VOLATILE_REL_PATHS
    )
    return {
        "artifact_protocol_version": manifest.artifact_protocol_version,
        "schema_version": manifest.schema_version,
        "model_fingerprint": manifest.model_fingerprint,
        "adapter_fingerprint": manifest.adapter_fingerprint,
        "config_fingerprint": manifest.config_fingerprint,
        "evaluation_fingerprint": manifest.evaluation_fingerprint,
        "export_backend_key": manifest.export_backend_key,
        "export_types": tuple(manifest.export_types),
        "export_fingerprint": manifest.export_fingerprint,
        "artifact_paths": tuple(sorted(manifest.artifact_paths)),
        "required_artifacts": tuple(sorted(manifest.required_artifacts)),
        "artifacts": artifacts,
        "training_protocol_version": manifest.training_protocol_version,
    }


def test_golden_export_determinism(tmp_path: Path) -> None:
    dir_a = tmp_path / "export_a"
    dir_b = tmp_path / "export_b"
    dir_a.mkdir()
    dir_b.mkdir()

    ctx_a = build_stub_export_context(output_dir=dir_a)
    ctx_b = build_stub_export_context(output_dir=dir_b)

    _, bundle_a = run_stub_export(ctx_a)
    _, bundle_b = run_stub_export(ctx_b)

    portable_a = _portable_manifest_fields(bundle_a.manifest)
    portable_b = _portable_manifest_fields(bundle_b.manifest)
    assert portable_a == portable_b

    assert bundle_a.manifest.export_fingerprint == bundle_b.manifest.export_fingerprint

    checksums_a = {
        d.relative_path: d.checksum
        for d in bundle_a.manifest.artifacts
        if d.relative_path not in {"export_manifest.json", "checksums.sha256"}
    }
    checksums_b = {
        d.relative_path: d.checksum
        for d in bundle_b.manifest.artifacts
        if d.relative_path not in {"export_manifest.json", "checksums.sha256"}
    }
    assert checksums_a == checksums_b
    assert tuple(sorted(checksums_a)) == tuple(sorted(checksums_b))
