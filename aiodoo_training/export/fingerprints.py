"""Deterministic export fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from aiodoo_training.domain.export_manifest import ArtifactDescriptor, ExportManifest


def canonical_json(data: Mapping[str, Any] | dict[str, Any]) -> str:
    """Portable sorted JSON serialization for fingerprint inputs."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    """Return sha256 hex digest of file contents."""
    from pathlib import Path

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_model_card_fingerprint(card_data: Mapping[str, Any]) -> str:
    """Fingerprint canonical model card JSON excluding volatile timestamps."""
    portable = {k: v for k, v in card_data.items() if k not in {"created_at", "updated_at"}}
    return sha256_hex(canonical_json(portable).encode("utf-8"))


def compute_manifest_metadata_fingerprint(manifest: ExportManifest) -> str:
    """Fingerprint selected ExportManifest fields excluding volatile paths/times."""
    payload = {
        "artifact_protocol_version": manifest.artifact_protocol_version,
        "schema_version": manifest.schema_version,
        "experiment_id": manifest.experiment_id.value,
        "run_id": manifest.run_id.value,
        "model_fingerprint": manifest.model_fingerprint,
        "adapter_fingerprint": manifest.adapter_fingerprint,
        "config_fingerprint": manifest.config_fingerprint,
        "evaluation_fingerprint": manifest.evaluation_fingerprint or "",
        "export_types": list(manifest.export_types),
        "export_backend_key": manifest.export_backend_key,
        "artifacts": [
            {
                "role": d.role,
                "relative_path": d.relative_path,
                "checksum": d.checksum,
                "required": d.required,
            }
            for d in manifest.artifacts
        ],
    }
    return sha256_hex(canonical_json(payload).encode("utf-8"))


def compute_export_fingerprint(
    *,
    model_fingerprint: str,
    adapter_fingerprint: str,
    config_fingerprint: str,
    evaluation_fingerprint: str | None,
    model_card_fingerprint: str,
    artifact_descriptors: Sequence[ArtifactDescriptor],
    export_types: Sequence[str],
    artifact_protocol_version: str,
) -> str:
    """
    Ordered concatenation digest for portable bundle identity.

    Absolute output paths must never be included.
    """
    parts = [
        f"protocol={artifact_protocol_version}",
        f"model={model_fingerprint}",
        f"adapter={adapter_fingerprint}",
        f"config={config_fingerprint}",
        f"evaluation={evaluation_fingerprint or ''}",
        f"model_card={model_card_fingerprint}",
        f"types={','.join(sorted(export_types))}",
    ]
    for descriptor in sorted(artifact_descriptors, key=lambda d: d.relative_path):
        parts.append(f"{descriptor.relative_path}:{descriptor.checksum}")
    material = "\n".join(parts).encode("utf-8")
    return sha256_hex(material)
