"""Build portable DistributedTopology from specs (no host noise in mesh_digest)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from aiodoo_training.distributed.mesh_digest import compute_mesh_digest
from aiodoo_training.domain.config import DistributedSpec
from aiodoo_training.domain.distributed_session import (
    DistributedTopology,
    Node,
    ProcessGroupHandle,
)


def build_rank_to_coord(world_size: int, mesh_shape: Sequence[int]) -> dict[int, tuple[int, ...]]:
    """Map ranks to mesh coordinates in deterministic row-major order."""
    if not mesh_shape:
        raise ValueError("mesh_shape must be non-empty.")
    product = 1
    for dim in mesh_shape:
        product *= int(dim)
    if product != world_size:
        raise ValueError(
            f"product(mesh_shape)={product} must equal world_size={world_size}."
        )
    mapping: dict[int, tuple[int, ...]] = {}
    for rank in range(world_size):
        coords: list[int] = []
        rem = rank
        for dim in reversed(mesh_shape):
            coords.append(rem % int(dim))
            rem //= int(dim)
        mapping[rank] = tuple(reversed(coords))
    return mapping


def build_topology(
    distributed: DistributedSpec,
    *,
    mesh_axes: Sequence[str],
    mesh_shape: Sequence[int],
    placement_key: str,
    communication_backend_key: str,
    accelerator: str,
    runtime_backend_key: str,
    node_id: str = "node-0",
) -> DistributedTopology:
    """
    Construct topology for this process.

    ``node_id`` may describe the local process for runtime use but is **not**
    included in ``mesh_digest``.
    """
    world_size = int(distributed.world_size)
    global_rank = int(distributed.global_rank)
    local_rank = int(distributed.local_rank)
    if world_size < 1:
        raise ValueError("world_size must be >= 1.")
    if not distributed.enabled:
        world_size = 1
        global_rank = 0
        local_rank = 0
        mesh_shape = (1,)
        mesh_axes = ("data",)

    coords = build_rank_to_coord(world_size, mesh_shape)
    digest = compute_mesh_digest(
        world_size=world_size,
        mesh_axes=tuple(mesh_axes),
        mesh_shape=tuple(int(s) for s in mesh_shape),
        placement_key=placement_key,
        communication_backend_key=communication_backend_key,
        accelerator=accelerator,
        runtime_backend_key=runtime_backend_key,
        rank_to_coord=coords,
    )
    ranks = tuple(range(world_size))
    groups: Mapping[str, ProcessGroupHandle] = {
        "default": ProcessGroupHandle(
            group_id="default",
            rank_set=ranks,
            backend_key=communication_backend_key,
        )
    }
    node = Node(
        node_id=node_id,
        local_ranks=(local_rank,),
        device_ids=(local_rank,),
    )
    return DistributedTopology(
        world_size=world_size,
        global_rank=global_rank,
        local_rank=local_rank,
        node_id=node_id,
        nodes=(node,),
        groups=groups,
        mesh_digest=digest,
    )
