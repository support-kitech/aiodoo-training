"""Phase 7 device mesh and placement plan domain DTOs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from aiodoo_training.domain.enums import DeviceKind


@dataclass(frozen=True, slots=True)
class MeshAxis:
    """Named mesh axis (e.g. data, model)."""

    name: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("MeshAxis.name must be non-empty.")


@dataclass(frozen=True, slots=True)
class DeviceMesh:
    """Immutable portable device mesh description."""

    mesh_id: str
    axes: tuple[MeshAxis, ...]
    shape: tuple[int, ...]
    device_kind: DeviceKind = DeviceKind.CPU
    rank_to_coord: Mapping[int, tuple[int, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.mesh_id or not self.mesh_id.strip():
            raise ValueError("DeviceMesh.mesh_id must be non-empty.")
        if len(self.axes) != len(self.shape):
            raise ValueError("DeviceMesh.axes and shape must have the same length.")
        if any(s < 1 for s in self.shape):
            raise ValueError("DeviceMesh.shape entries must be >= 1.")
        object.__setattr__(
            self,
            "rank_to_coord",
            MappingProxyType(
                {int(k): tuple(int(x) for x in v) for k, v in self.rank_to_coord.items()}
            ),
        )


@dataclass(frozen=True, slots=True)
class PlacementPlan:
    """Immutable rank → device / shard role assignment."""

    strategy_key: str
    world_size: int
    rank_to_device_id: Mapping[int, int] = field(default_factory=lambda: MappingProxyType({}))
    rank_roles: Mapping[int, str] = field(default_factory=lambda: MappingProxyType({}))
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.strategy_key or not self.strategy_key.strip():
            raise ValueError("PlacementPlan.strategy_key must be non-empty.")
        if self.world_size < 1:
            raise ValueError("PlacementPlan.world_size must be >= 1.")
        object.__setattr__(
            self,
            "rank_to_device_id",
            MappingProxyType({int(k): int(v) for k, v in self.rank_to_device_id.items()}),
        )
        object.__setattr__(
            self,
            "rank_roles",
            MappingProxyType({int(k): str(v) for k, v in self.rank_roles.items()}),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType({str(k): str(v) for k, v in self.metadata.items()}),
        )
