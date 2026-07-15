"""ArtifactIndex load/save for cross-bundle discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from aiodoo_training.domain.export_manifest import ArtifactIndexEntry
from aiodoo_training.domain.identifiers import ExperimentId, RunId

INDEX_FILENAME = "artifacts.json"


@dataclass(frozen=True, slots=True)
class ArtifactIndex:
    """Cross-bundle locator under an export output root."""

    output_dir: Path
    entries: tuple[ArtifactIndexEntry, ...] = ()

    @property
    def index_path(self) -> Path:
        return self.output_dir / INDEX_FILENAME

    def append(self, entry: ArtifactIndexEntry) -> ArtifactIndex:
        return replace(self, entries=self.entries + (entry,))

    def upsert(self, entry: ArtifactIndexEntry) -> ArtifactIndex:
        filtered = tuple(e for e in self.entries if e.bundle_path != entry.bundle_path)
        return replace(self, entries=filtered + (entry,))

    def save(self, path: Path | None = None) -> Path:
        target = path if path is not None else self.index_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1",
            "entries": [_entry_to_dict(e) for e in self.entries],
        }
        target.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, output_dir: Path) -> ArtifactIndex:
        path = output_dir / INDEX_FILENAME
        if not path.is_file():
            return cls(output_dir=output_dir, entries=())
        data = json.loads(path.read_text(encoding="utf-8"))
        entries_raw = data.get("entries") or []
        entries = tuple(_entry_from_dict(item) for item in entries_raw)
        return cls(output_dir=output_dir, entries=entries)


def _entry_to_dict(entry: ArtifactIndexEntry) -> dict[str, object]:
    return {
        "bundle_path": entry.bundle_path,
        "experiment_id": entry.experiment_id.value,
        "run_id": entry.run_id.value,
        "export_fingerprint": entry.export_fingerprint,
        "artifact_protocol_version": entry.artifact_protocol_version,
        "export_types": list(entry.export_types),
        "roles": list(entry.roles),
        "manifest_relpath": entry.manifest_relpath,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
    }


def _entry_from_dict(data: dict[str, object]) -> ArtifactIndexEntry:
    created_raw = data.get("created_at")
    created_at = datetime.fromisoformat(str(created_raw)) if created_raw else None
    export_types_raw = data.get("export_types")
    roles_raw = data.get("roles")
    export_types: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    if isinstance(export_types_raw, list):
        export_types = tuple(str(t) for t in export_types_raw)
    if isinstance(roles_raw, list):
        roles = tuple(str(r) for r in roles_raw)
    return ArtifactIndexEntry(
        bundle_path=str(data["bundle_path"]),
        experiment_id=ExperimentId(value=str(data["experiment_id"])),
        run_id=RunId(value=str(data["run_id"])),
        export_fingerprint=str(data["export_fingerprint"]),
        artifact_protocol_version=str(data["artifact_protocol_version"]),
        export_types=export_types,
        roles=roles,
        manifest_relpath=str(data.get("manifest_relpath") or "export_manifest.json"),
        created_at=created_at,
    )
