"""Phase 6 artifact lineage domain (observational indexes only)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from aiodoo_training.domain.enums import ArtifactRelationKind
from aiodoo_training.domain.identifiers import RunId


@dataclass(frozen=True, slots=True)
class ArtifactVersion:
    """Local versioning label / digest pointer."""

    label: str
    digest: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class ArtifactRelationship:
    """Typed edge between runs / artifacts."""

    kind: ArtifactRelationKind
    source: str
    target: str
    run_id: RunId | None = None


@dataclass(frozen=True, slots=True)
class ArtifactLineage:
    """Graph edges for observed artifacts."""

    relationships: tuple[ArtifactRelationship, ...] = ()

    def with_edge(self, edge: ArtifactRelationship) -> ArtifactLineage:
        return replace(self, relationships=(*self.relationships, edge))


@dataclass(frozen=True, slots=True)
class ArtifactHistoryEntry:
    """One observed artifact pointer."""

    path: str
    role: str
    digest: str = ""
    run_id: RunId | None = None
