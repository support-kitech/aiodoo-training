"""Phase 7 distributed readiness unit tests (CPU / fake backend only)."""

from __future__ import annotations

import pytest

from aiodoo_training.bootstrap import bootstrap_phase7
from aiodoo_training.config.distributed_config import (
    parse_phase7_distributed_config,
    to_distributed_spec,
    to_runtime_policy,
    validate_phase7_distributed_fragments,
)
from aiodoo_training.distributed.coordinator import (
    DistributedCheckpointCoordinator,
    DistributedCoordinator,
)
from aiodoo_training.distributed.fault_tolerance import FaultToleranceCoordinator
from aiodoo_training.distributed.mesh_digest import compute_mesh_digest
from aiodoo_training.distributed.placement import DistributedPlacementResolver
from aiodoo_training.distributed.runtime import DistributedRuntime
from aiodoo_training.distributed.seed import derive_rank_seed
from aiodoo_training.distributed.shard_planner import ShardPlanner
from aiodoo_training.domain.config import DistributedSpec, ExecutionSpec
from aiodoo_training.domain.distributed_health import DistributedHealth
from aiodoo_training.domain.distributed_policies import (
    RestartPolicy,
)
from aiodoo_training.domain.enums import (
    ClusterStatus,
    DistributedStatus,
    RankRole,
    ReductionOp,
    RestartFrom,
)
from aiodoo_training.domain.resources import (
    ExecutionEnvironment,
)
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.exceptions import ConfigError, DistributedError
from aiodoo_training.factories import (
    DistributedBackendFactory,
    DistributedSamplerFactory,
    PlacementStrategyFactory,
)
from aiodoo_training.infrastructure.resources import StaticResourcePlanner


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase7(overwrite=True)


def _cpu_env() -> ExecutionEnvironment:
    return StaticResourcePlanner().resolve_spec(ExecutionSpec())


def test_registry_keys() -> None:
    for key in ("fake", "ddp", "fsdp", "deepspeed", "accelerate", "xla"):
        assert DistributedBackendFactory().create(key) is not None
    assert PlacementStrategyFactory().create("single") is not None
    assert DistributedSamplerFactory().create("shard") is not None


def test_fake_backend_collectives() -> None:
    env = _cpu_env()
    policy = to_runtime_policy(
        parse_phase7_distributed_config(
            {"enabled": True, "backend": "fake", "world_size": 1, "topology": {"mesh_shape": [1]}}
        )
    )
    runtime = DistributedRuntime()
    ctx = runtime.open(policy, env, to_distributed_spec(parse_phase7_distributed_config(
        {"enabled": True, "world_size": 1, "topology": {"mesh_shape": [1]}}
    )))
    assert ctx.session.status is DistributedStatus.READY
    sync = runtime.sync
    assert sync is not None
    sync.barrier()
    reduced = sync.all_reduce({"loss": 2.0}, op=ReductionOp.MEAN)
    assert reduced["loss"] == 2.0
    runtime.close()


def test_mesh_digest_portable_and_stable() -> None:
    a = compute_mesh_digest(
        world_size=2,
        mesh_axes=("data",),
        mesh_shape=(2,),
        placement_key="data_parallel",
        communication_backend_key="fake",
        accelerator="none",
        runtime_backend_key="fake",
        rank_to_coord={0: (0,), 1: (1,)},
    )
    b = compute_mesh_digest(
        world_size=2,
        mesh_axes=("data",),
        mesh_shape=(2,),
        placement_key="data_parallel",
        communication_backend_key="fake",
        accelerator="none",
        runtime_backend_key="fake",
        rank_to_coord={0: (0,), 1: (1,)},
    )
    assert a == b
    assert "localhost" not in a
    assert len(a) == 32  # MESH_DIGEST_LENGTH; value unchanged by named constant


def test_placement_resolver() -> None:
    env = _cpu_env()
    mesh, plan, digest = DistributedPlacementResolver().resolve(
        env,
        DistributedSpec(enabled=True, world_size=2, global_rank=0, local_rank=0),
        placement_key="data_parallel",
        mesh_axes=("data",),
        mesh_shape=(2,),
        communication_backend_key="fake",
        runtime_backend_key="fake",
    )
    assert mesh.shape == (2,)
    assert plan.world_size == 2
    assert digest == mesh.digest


def test_shard_planner_reuses_dataset_session_fields() -> None:
    env = _cpu_env()
    policy = to_runtime_policy(
        parse_phase7_distributed_config(
            {
                "enabled": True,
                "world_size": 4,
                "global_rank": 2,
                "local_rank": 2,
                "topology": {"placement": "data_parallel", "mesh_shape": [4]},
            }
        )
    )
    runtime = DistributedRuntime()
    ctx = runtime.open(
        policy,
        env,
        to_distributed_spec(
            parse_phase7_distributed_config(
                {
                    "enabled": True,
                    "world_size": 4,
                    "global_rank": 2,
                    "local_rank": 2,
                    "topology": {"mesh_shape": [4]},
                }
            )
        ),
    )
    session = DatasetSession(session_id="s1")
    updated = ShardPlanner().apply(session, ctx.session.topology)
    assert updated.world_size == 4
    assert updated.global_rank == 2
    assert updated.shard_id == 2
    assert updated.num_shards == 4
    runtime.close()


def test_sampler_determinism() -> None:
    sampler = DistributedSamplerFactory().create("shard")
    s0 = DatasetSession(session_id="s", world_size=2, global_rank=0, shard_id=0, num_shards=2)
    s1 = DatasetSession(session_id="s", world_size=2, global_rank=1, shard_id=1, num_shards=2)
    a = list(sampler.sample_indices(10, s0, seed=7))
    b = list(sampler.sample_indices(10, s0, seed=7))
    c = list(sampler.sample_indices(10, s1, seed=7))
    assert a == b
    assert set(a).isdisjoint(set(c))
    assert len(a) + len(c) == 10


def test_checkpoint_coordinator_roles() -> None:
    env = _cpu_env()
    frag = parse_phase7_distributed_config(
        {"enabled": True, "world_size": 2, "topology": {"mesh_shape": [2]}}
    )
    runtime = DistributedRuntime()
    ctx = runtime.open(to_runtime_policy(frag), env, to_distributed_spec(frag))
    sync = runtime.sync
    assert sync is not None
    roles = DistributedCoordinator()
    assert roles.role_for(ctx, "checkpoint") in {RankRole.COORDINATOR, RankRole.IDLE}
    coord = DistributedCheckpointCoordinator(sync, roles)
    role = coord.prepare_save(ctx)
    assert coord.may_invoke_checkpoint_manager(role) or role is RankRole.IDLE
    coord.finish_save(ctx)
    runtime.close()


def test_restart_policy_does_not_ignore_mesh_mismatch() -> None:
    ft = FaultToleranceCoordinator(
        RestartPolicy(max_restarts=2, require_same_mesh_digest=True)
    )
    health = DistributedHealth(cluster=ClusterStatus.FAILED, message="worker lost")
    decision = ft.on_incident(
        health,
        current_world_size=2,
        checkpoint_world_size=2,
        current_mesh_digest="aaa",
        checkpoint_mesh_digest="bbb",
    )
    assert decision.should_restart is False
    assert "mesh_digest" in decision.reason


def test_restart_allows_when_compatible() -> None:
    ft = FaultToleranceCoordinator(RestartPolicy(max_restarts=1))
    health = DistributedHealth(cluster=ClusterStatus.FAILED, message="timeout")
    decision = ft.on_incident(
        health,
        current_world_size=1,
        checkpoint_world_size=1,
        current_mesh_digest="x",
        checkpoint_mesh_digest="x",
    )
    assert decision.should_restart is True
    assert decision.restart_from is RestartFrom.LAST_CKPT


def test_config_mesh_shape_validation() -> None:
    with pytest.raises(ConfigError):
        validate_phase7_distributed_fragments(
            parse_phase7_distributed_config(
                {"enabled": True, "world_size": 4, "topology": {"mesh_shape": [2]}}
            )
        )


def test_registration_only_backend_fails_init() -> None:
    backend = DistributedBackendFactory().create("ddp")
    from aiodoo_training.distributed.topology import build_topology

    topo = build_topology(
        DistributedSpec(enabled=False),
        mesh_axes=("data",),
        mesh_shape=(1,),
        placement_key="single",
        communication_backend_key="fake",
        accelerator="none",
        runtime_backend_key="ddp",
    )
    with pytest.raises(DistributedError, match="registration-only"):
        backend.initialize(topo)


def test_rank_seed_stable() -> None:
    assert derive_rank_seed(42, 0) == derive_rank_seed(42, 0)
    assert derive_rank_seed(42, 0) != derive_rank_seed(42, 1)


def test_distributed_health_orthogonal_to_tracking() -> None:
    from aiodoo_training.domain.enums import TrackingHealthStatus
    from aiodoo_training.domain.tracking_policies import TrackingHealth

    dh = DistributedHealth(cluster=ClusterStatus.HEALTHY)
    th = TrackingHealth(backend_key="null", status=TrackingHealthStatus.HEALTHY)
    assert dh.cluster is ClusterStatus.HEALTHY
    assert th.status is TrackingHealthStatus.HEALTHY
    assert type(dh) is not type(th)
