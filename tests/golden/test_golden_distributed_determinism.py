"""Golden determinism for Phase 7 fake distributed path."""

from __future__ import annotations

import pytest

from aiodoo_training.bootstrap import bootstrap_phase7
from aiodoo_training.config.distributed_config import (
    parse_phase7_distributed_config,
    to_distributed_spec,
    to_runtime_policy,
)
from aiodoo_training.distributed.mesh_digest import compute_mesh_digest
from aiodoo_training.distributed.runtime import DistributedRuntime
from aiodoo_training.distributed.seed import derive_rank_seed
from aiodoo_training.domain.config import ExecutionSpec
from aiodoo_training.domain.enums import ReductionOp
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.factories import DistributedSamplerFactory
from aiodoo_training.infrastructure.resources import StaticResourcePlanner


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase7(overwrite=True)


def test_golden_mesh_digest() -> None:
    digest = compute_mesh_digest(
        world_size=4,
        mesh_axes=("data",),
        mesh_shape=(4,),
        placement_key="data_parallel",
        communication_backend_key="fake",
        accelerator="none",
        runtime_backend_key="fake",
        rank_to_coord={0: (0,), 1: (1,), 2: (2,), 3: (3,)},
    )
    assert digest == compute_mesh_digest(
        world_size=4,
        mesh_axes=("data",),
        mesh_shape=(4,),
        placement_key="data_parallel",
        communication_backend_key="fake",
        accelerator="none",
        runtime_backend_key="fake",
        rank_to_coord={0: (0,), 1: (1,), 2: (2,), 3: (3,)},
    )


def test_golden_sampler_indices() -> None:
    sampler = DistributedSamplerFactory().create("shard")
    session = DatasetSession(
        session_id="gold", world_size=2, global_rank=0, shard_id=0, num_shards=2, epoch=0
    )
    indices = list(sampler.sample_indices(16, session, seed=123))
    assert indices == list(sampler.sample_indices(16, session, seed=123))


def test_golden_fake_reduce_and_session() -> None:
    env = StaticResourcePlanner().resolve_spec(ExecutionSpec())
    frag = parse_phase7_distributed_config(
        {"enabled": True, "backend": "fake", "world_size": 1, "topology": {"mesh_shape": [1]}}
    )
    runtime = DistributedRuntime()
    ctx = runtime.open(to_runtime_policy(frag), env, to_distributed_spec(frag))
    sync = runtime.sync
    assert sync is not None
    a = sync.all_reduce({"loss": 1.5, "acc": 0.5}, op=ReductionOp.MEAN)
    b = sync.all_reduce({"loss": 1.5, "acc": 0.5}, op=ReductionOp.MEAN)
    assert a == b
    assert ctx.session.topology.mesh_digest == ctx.mesh.digest
    runtime.close()


def test_golden_rank_seed() -> None:
    assert derive_rank_seed(99, 3, epoch=2) == derive_rank_seed(99, 3, epoch=2)
