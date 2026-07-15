"""PlacementStrategy implementations (framework-free)."""

from __future__ import annotations

from types import MappingProxyType

from aiodoo_training.domain.device_mesh import DeviceMesh, PlacementPlan
from aiodoo_training.domain.distributed_session import DistributedTopology
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.ports.distributed import PlacementStrategy


class SinglePlacement(PlacementStrategy):
    def place(
        self,
        env: ExecutionEnvironment,
        topology: DistributedTopology,
        *,
        mesh_spec: DeviceMesh | None = None,
    ) -> PlacementPlan:
        del env, mesh_spec
        ws = topology.world_size
        return PlacementPlan(
            strategy_key="single",
            world_size=ws,
            rank_to_device_id=MappingProxyType({r: 0 for r in range(ws)}),
            rank_roles=MappingProxyType({r: "worker" for r in range(ws)}),
        )


class DataParallelPlacement(PlacementStrategy):
    def place(
        self,
        env: ExecutionEnvironment,
        topology: DistributedTopology,
        *,
        mesh_spec: DeviceMesh | None = None,
    ) -> PlacementPlan:
        del env
        ws = topology.world_size
        devices = {
            r: (mesh_spec.rank_to_coord[r][-1] if mesh_spec and r in mesh_spec.rank_to_coord else r)
            for r in range(ws)
        }
        return PlacementPlan(
            strategy_key="data_parallel",
            world_size=ws,
            rank_to_device_id=MappingProxyType(devices),
            rank_roles=MappingProxyType({r: "data_parallel" for r in range(ws)}),
        )


class FsdpAutoPlacement(PlacementStrategy):
    def place(
        self,
        env: ExecutionEnvironment,
        topology: DistributedTopology,
        *,
        mesh_spec: DeviceMesh | None = None,
    ) -> PlacementPlan:
        base = DataParallelPlacement().place(env, topology, mesh_spec=mesh_spec)
        return PlacementPlan(
            strategy_key="fsdp_auto",
            world_size=base.world_size,
            rank_to_device_id=base.rank_to_device_id,
            rank_roles=MappingProxyType({r: "fsdp" for r in range(base.world_size)}),
            metadata=MappingProxyType({"hint": "fsdp_auto"}),
        )


class DeepspeedZeroPlacement(PlacementStrategy):
    def place(
        self,
        env: ExecutionEnvironment,
        topology: DistributedTopology,
        *,
        mesh_spec: DeviceMesh | None = None,
    ) -> PlacementPlan:
        base = DataParallelPlacement().place(env, topology, mesh_spec=mesh_spec)
        return PlacementPlan(
            strategy_key="deepspeed_zero",
            world_size=base.world_size,
            rank_to_device_id=base.rank_to_device_id,
            rank_roles=MappingProxyType({r: "zero" for r in range(base.world_size)}),
            metadata=MappingProxyType({"hint": "deepspeed_zero"}),
        )


def register_default_placement_strategies(*, overwrite: bool = False) -> None:
    from aiodoo_training.registries import placement_strategy_registry

    placement_strategy_registry.register("single", SinglePlacement, overwrite=overwrite)
    placement_strategy_registry.register(
        "data_parallel", DataParallelPlacement, overwrite=overwrite
    )
    placement_strategy_registry.register("fsdp_auto", FsdpAutoPlacement, overwrite=overwrite)
    placement_strategy_registry.register(
        "deepspeed_zero", DeepspeedZeroPlacement, overwrite=overwrite
    )
