"""DistributedPlacementResolver — companion to frozen ResourcePlanner."""

from __future__ import annotations

from collections.abc import Sequence

from aiodoo_training.distributed.mesh_digest import compute_mesh_digest
from aiodoo_training.distributed.topology import build_rank_to_coord, build_topology
from aiodoo_training.domain.config import DistributedSpec
from aiodoo_training.domain.device_mesh import DeviceMesh, MeshAxis, PlacementPlan
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.factories.factories import PlacementStrategyFactory
from aiodoo_training.ports.distributed import PlacementStrategy


class DistributedPlacementResolver:
    """
    Consumes ExecutionEnvironment + DistributedSpec; does not redesign ResourcePlanner.
    """

    def __init__(self, strategy_factory: PlacementStrategyFactory | None = None) -> None:
        self._strategies = strategy_factory or PlacementStrategyFactory()

    def resolve(
        self,
        env: ExecutionEnvironment,
        distributed: DistributedSpec,
        *,
        placement_key: str = "single",
        mesh_axes: Sequence[str] = ("data",),
        mesh_shape: Sequence[int] = (1,),
        communication_backend_key: str = "fake",
        runtime_backend_key: str = "fake",
    ) -> tuple[DeviceMesh, PlacementPlan, str]:
        world_size = 1 if not distributed.enabled else int(distributed.world_size)
        axes = tuple(mesh_axes) if mesh_axes else ("data",)
        shape = tuple(int(s) for s in mesh_shape) if mesh_shape else (world_size,)
        if not distributed.enabled:
            shape = (1,)
            axes = ("data",)
        product = 1
        for dim in shape:
            product *= dim
        if product != world_size:
            raise ValueError(
                f"product(mesh_shape)={product} must equal world_size={world_size}."
            )
        coords = build_rank_to_coord(world_size, shape)
        digest = compute_mesh_digest(
            world_size=world_size,
            mesh_axes=axes,
            mesh_shape=shape,
            placement_key=placement_key,
            communication_backend_key=communication_backend_key,
            accelerator=env.accelerator.value,
            runtime_backend_key=runtime_backend_key,
            rank_to_coord=coords,
        )
        mesh = DeviceMesh(
            mesh_id=f"mesh-{digest[:12]}",
            axes=tuple(MeshAxis(name=a) for a in axes),
            shape=shape,
            device_kind=env.selected_device,
            rank_to_coord=coords,
            digest=digest,
        )
        topology = build_topology(
            DistributedSpec(
                enabled=distributed.enabled,
                world_size=world_size,
                global_rank=0 if not distributed.enabled else distributed.global_rank,
                local_rank=0 if not distributed.enabled else distributed.local_rank,
                num_nodes=distributed.num_nodes,
            ),
            mesh_axes=axes,
            mesh_shape=shape,
            placement_key=placement_key,
            communication_backend_key=communication_backend_key,
            accelerator=env.accelerator.value,
            runtime_backend_key=runtime_backend_key,
        )
        strategy: PlacementStrategy = self._strategies.create(placement_key)
        plan = strategy.place(env, topology, mesh_spec=mesh)
        return mesh, plan, digest
