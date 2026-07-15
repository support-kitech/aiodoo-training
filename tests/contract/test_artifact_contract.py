"""Contract tests for published artifact bundles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase4
from aiodoo_training.domain.export_manifest import (
    ARTIFACT_PROTOCOL_VERSION,
    ArtifactCompatibilityPolicy,
    ArtifactDescriptor,
    ExportManifest,
)
from aiodoo_training.export import ArtifactIndex, run_stub_export
from aiodoo_training.export.fingerprints import sha256_file


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase4(overwrite=True)


def _validate_descriptor_checksums(
    bundle_root: Path, descriptors: tuple[ArtifactDescriptor, ...]
) -> None:
    for descriptor in descriptors:
        if descriptor.relative_path in {"export_manifest.json", "checksums.sha256"}:
            continue
        file_path = bundle_root / descriptor.relative_path
        assert file_path.is_file(), f"Missing artifact file: {descriptor.relative_path}"
        actual = sha256_file(str(file_path))
        assert actual == descriptor.checksum, (
            f"Checksum mismatch for {descriptor.relative_path}: "
            f"expected {descriptor.checksum}, got {actual}"
        )


def _validate_compatibility(
    manifest: ExportManifest,
    policy: ArtifactCompatibilityPolicy,
) -> None:
    assert manifest.artifact_protocol_version in policy.accepted_artifact_protocols
    present_roles = {d.role for d in manifest.artifacts}
    for role in policy.required_roles:
        if role in {"manifest", "model_card"}:
            continue
        assert role in present_roles or role in manifest.export_types, (
            f"Required role {role!r} missing from bundle"
        )


def test_artifact_bundle_contract(tmp_path: Path) -> None:
    _, bundle = run_stub_export(output_dir=tmp_path)

    manifest_path = bundle.root / "export_manifest.json"
    assert manifest_path.is_file()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ExportManifest.from_dict(manifest_data)

    assert manifest.artifact_protocol_version == ARTIFACT_PROTOCOL_VERSION
    _validate_descriptor_checksums(bundle.root, manifest.artifacts)

    index = ArtifactIndex.load(tmp_path)
    assert len(index.entries) >= 1
    entry = next(e for e in index.entries if e.export_fingerprint == manifest.export_fingerprint)
    assert entry.artifact_protocol_version == ARTIFACT_PROTOCOL_VERSION
    assert (tmp_path / entry.bundle_path).resolve() == bundle.root.resolve()

    policy = ArtifactCompatibilityPolicy(
        accepted_artifact_protocols=(ARTIFACT_PROTOCOL_VERSION,),
        required_roles=("peft_adapter", "manifest"),
        optional_roles=("tokenizer", "model_card", "evaluation_report"),
    )
    _validate_compatibility(manifest, policy)
