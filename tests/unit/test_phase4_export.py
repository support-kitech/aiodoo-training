"""Phase 4 export unit tests — manager, manifest, index, compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase4
from aiodoo_training.builders import ExportBuilder
from aiodoo_training.domain.export_manifest import (
    ARTIFACT_PROTOCOL_VERSION,
    ArtifactCompatibilityPolicy,
    ArtifactValidationPolicy,
    ExportManifest,
)
from aiodoo_training.export import (
    ArtifactIndex,
    ExportManager,
    build_stub_export_context,
    run_stub_export,
)
from aiodoo_training.export.fingerprints import sha256_file
from aiodoo_training.factories import ExporterFactory
from aiodoo_training.infrastructure.stub.exporter import StubExporter


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase4(overwrite=True)


def test_export_manager_publishes_bundle(tmp_path: Path) -> None:
    ctx = build_stub_export_context(output_dir=tmp_path)
    _, bundle = ExportManager().export(ctx)
    assert bundle.root.is_dir()
    assert bundle.root.parent == tmp_path
    assert (bundle.root / "export_manifest.json").is_file()


def test_manifest_artifact_protocol_version(tmp_path: Path) -> None:
    _, bundle = run_stub_export(output_dir=tmp_path)
    assert bundle.manifest.artifact_protocol_version == "1"
    assert bundle.manifest.artifact_protocol_version == ARTIFACT_PROTOCOL_VERSION


def test_checksums_file_exists(tmp_path: Path) -> None:
    _, bundle = run_stub_export(output_dir=tmp_path)
    checksums = bundle.root / "checksums.sha256"
    assert checksums.is_file()
    lines = checksums.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert len(digest) == 64
        assert (bundle.root / rel).is_file()


def test_artifact_index_updated(tmp_path: Path) -> None:
    _, bundle = run_stub_export(output_dir=tmp_path)
    index = ArtifactIndex.load(tmp_path)
    assert len(index.entries) == 1
    entry = index.entries[0]
    assert entry.export_fingerprint == bundle.manifest.export_fingerprint
    assert entry.artifact_protocol_version == "1"
    assert (tmp_path / entry.bundle_path).resolve() == bundle.root.resolve()


def test_model_card_written_when_requested(tmp_path: Path) -> None:
    ctx = build_stub_export_context(output_dir=tmp_path)
    _, bundle = ExportManager().export(ctx)
    card_md = bundle.root / "model_card.md"
    card_json = bundle.root / "model_card.json"
    assert card_md.is_file()
    assert card_json.is_file()
    roles = {d.role for d in bundle.manifest.artifacts}
    assert "model_card" in roles


def test_artifact_validation_and_compatibility_policies() -> None:
    builder = (
        ExportBuilder()
        .with_backend("stub")
        .with_export_types("peft_adapter", "manifest", "model_card")
        .with_validation_policy(ArtifactValidationPolicy.WARN)
    )
    spec = builder.build_spec()
    assert "peft_adapter" in spec.export_types
    assert builder.validation_policy is ArtifactValidationPolicy.WARN

    policy = ArtifactCompatibilityPolicy(
        accepted_artifact_protocols=("1",),
        required_roles=("peft_adapter", "manifest"),
        optional_roles=("model_card",),
    )
    assert ARTIFACT_PROTOCOL_VERSION in policy.accepted_artifact_protocols


def test_exporter_factory_creates_stub() -> None:
    exporter = ExporterFactory().create("stub")
    assert isinstance(exporter, StubExporter)
    assert exporter.BACKEND_KEY == "stub"


def test_manifest_checksums_match_disk(tmp_path: Path) -> None:
    _, bundle = run_stub_export(output_dir=tmp_path)
    manifest_path = bundle.root / "export_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ExportManifest.from_dict(data)
    for descriptor in manifest.artifacts:
        file_path = bundle.root / descriptor.relative_path
        if not file_path.is_file():
            continue
        if descriptor.relative_path in {"export_manifest.json", "checksums.sha256"}:
            continue
        assert sha256_file(str(file_path)) == descriptor.checksum


def test_export_manifest_serializes_dataset_version(tmp_path: Path) -> None:
    from aiodoo_training.builders.export_builders import ExportContextBuilder

    ctx = build_stub_export_context(output_dir=tmp_path)
    export_ctx = (
        ExportContextBuilder()
        .with_config(ctx.config)
        .with_piece("execution", ctx.execution)
        .with_piece("model", ctx.model)
        .with_piece("exporter", ctx.exporter)
        .with_piece("export_session", ctx.export_session)
        .with_piece("output_dir", ctx.output_dir)
        .with_piece("exporter_backend_key", ctx.exporter_backend_key)
        .with_piece("model_fingerprint", ctx.model_fingerprint)
        .with_piece("adapter_fingerprint", ctx.adapter_fingerprint)
        .with_piece("config_fingerprint", ctx.config_fingerprint)
        .with_piece("export_types", ctx.export_types)
        .with_piece("bind_extra", {"dataset_version": "v1.0.0"})
        .build()
    )
    _, bundle = ExportManager().export(export_ctx)
    manifest_path = bundle.root / "export_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data.get("dataset_version") == "v1.0.0"
    roundtrip = ExportManifest.from_dict(data)
    assert roundtrip.dataset_version == "v1.0.0"
    assert bundle.manifest.dataset_version == "v1.0.0"
