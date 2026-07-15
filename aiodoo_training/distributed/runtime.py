"""DistributedRuntime — owns PG lifecycle and health snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType
from uuid import uuid4

from aiodoo_training.distributed.context import DistributedContext
from aiodoo_training.distributed.placement import DistributedPlacementResolver
from aiodoo_training.distributed.sync import SyncFacade
from aiodoo_training.distributed.topology import build_topology
from aiodoo_training.domain.config import DistributedSpec
from aiodoo_training.domain.distributed_health import DistributedHealth
from aiodoo_training.domain.distributed_policies import DistributedRuntimePolicy
from aiodoo_training.domain.distributed_session import DistributedSession
from aiodoo_training.domain.enums import (
    ClusterStatus,
    DistributedStatus,
    NodeStatus,
    WorkerStatus,
)
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.exceptions import DistributedError, DistributedLifecycleError
from aiodoo_training.factories.factories import DistributedBackendFactory
from aiodoo_training.ports.distributed import DistributedBackend

# Legal DistributedSession transitions (see phase7 architecture §1.4).
_ALLOWED: dict[DistributedStatus, frozenset[DistributedStatus]] = {
    DistributedStatus.PENDING: frozenset(
        {DistributedStatus.INITIALIZING, DistributedStatus.ABORTED}
    ),
    DistributedStatus.INITIALIZING: frozenset({DistributedStatus.READY, DistributedStatus.FAILED}),
    DistributedStatus.READY: frozenset(
        {DistributedStatus.RUNNING, DistributedStatus.FAILED, DistributedStatus.ABORTED}
    ),
    DistributedStatus.RUNNING: frozenset(
        {
            DistributedStatus.DRAINING,
            DistributedStatus.FAILED,
            DistributedStatus.COMPLETED,
        }
    ),
    DistributedStatus.DRAINING: frozenset({DistributedStatus.COMPLETED, DistributedStatus.FAILED}),
    DistributedStatus.FAILED: frozenset(
        {DistributedStatus.INITIALIZING, DistributedStatus.ABORTED}
    ),
    DistributedStatus.COMPLETED: frozenset(),
    DistributedStatus.ABORTED: frozenset(),
}


def _transition(session: DistributedSession, to: DistributedStatus) -> DistributedSession:
    allowed = _ALLOWED.get(session.status, frozenset())
    if to not in allowed:
        raise DistributedLifecycleError(
            f"Illegal DistributedSession transition {session.status.value} → {to.value}"
        )
    return session.with_status(to)


class DistributedRuntime:
    """Open / use / close distributed execution resources."""

    def __init__(
        self,
        *,
        backend_factory: DistributedBackendFactory | None = None,
        placement: DistributedPlacementResolver | None = None,
    ) -> None:
        self._backends = backend_factory or DistributedBackendFactory()
        self._placement = placement or DistributedPlacementResolver()
        self._ctx: DistributedContext | None = None
        self._sync: SyncFacade | None = None

    @property
    def context(self) -> DistributedContext | None:
        return self._ctx

    @property
    def sync(self) -> SyncFacade | None:
        return self._sync

    def open(
        self,
        policy: DistributedRuntimePolicy,
        execution: ExecutionEnvironment,
        distributed: DistributedSpec,
    ) -> DistributedContext:
        if self._ctx is not None:
            raise DistributedError("DistributedRuntime already open.")
        backend_key = policy.backend_key if policy.enabled else "fake"
        placement_key = policy.placement_key if policy.enabled else "single"
        mesh_shape = policy.mesh_shape if policy.enabled else (1,)
        mesh_axes = policy.mesh_axes if policy.enabled else ("data",)
        if not policy.enabled:
            distributed = DistributedSpec(
                enabled=False, world_size=1, global_rank=0, local_rank=0, num_nodes=1
            )

        mesh, plan, _digest = self._placement.resolve(
            execution,
            distributed,
            placement_key=placement_key,
            mesh_axes=mesh_axes,
            mesh_shape=mesh_shape,
            communication_backend_key=policy.communication.key,
            runtime_backend_key=backend_key,
        )
        topology = build_topology(
            distributed,
            mesh_axes=mesh_axes,
            mesh_shape=mesh_shape,
            placement_key=placement_key,
            communication_backend_key=policy.communication.key,
            accelerator=execution.accelerator.value,
            runtime_backend_key=backend_key,
        )
        # Keep mesh.digest (already portable); topology.mesh_digest matches.
        session = DistributedSession(
            session_id=f"dist-{uuid4().hex[:12]}",
            topology=topology,
            runtime_backend_key=backend_key,
            status=DistributedStatus.PENDING,
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session = _transition(session, DistributedStatus.INITIALIZING)
        backend: DistributedBackend = self._backends.create(backend_key)
        try:
            backend.initialize(topology)
            session = _transition(session, DistributedStatus.READY)
            health = DistributedHealth(
                cluster=ClusterStatus.HEALTHY,
                workers=MappingProxyType(
                    {r: WorkerStatus.HEALTHY for r in range(topology.world_size)}
                ),
                nodes=MappingProxyType({topology.node_id: NodeStatus.HEALTHY}),
                message=None,
            )
        except Exception as exc:  # noqa: BLE001
            session = _transition(session, DistributedStatus.FAILED)
            self._ctx = DistributedContext(
                session=session,
                execution=execution,
                policy=policy,
                mesh=mesh,
                placement=plan,
                backend=backend,
                health=DistributedHealth(cluster=ClusterStatus.FAILED, message=str(exc)),
            )
            raise DistributedError(f"DistributedRuntime init failed: {exc}") from exc

        self._ctx = DistributedContext(
            session=session,
            execution=execution,
            policy=policy,
            mesh=mesh,
            placement=plan,
            backend=backend,
            health=health,
        )
        self._sync = SyncFacade(
            backend,
            default_group=policy.collective.default_group,
            require_deterministic_order=policy.collective.require_deterministic_order,
        )
        return self._ctx

    def mark_running(self) -> DistributedContext:
        ctx = self._require()
        session = _transition(ctx.session, DistributedStatus.RUNNING)
        self._ctx = ctx.with_session(session)
        return self._ctx

    def close(self, *, failed: bool = False) -> None:
        if self._ctx is None:
            return
        ctx = self._ctx
        try:
            if failed:
                if ctx.session.status not in {
                    DistributedStatus.FAILED,
                    DistributedStatus.ABORTED,
                    DistributedStatus.COMPLETED,
                }:
                    try:
                        session = _transition(ctx.session, DistributedStatus.FAILED)
                    except DistributedLifecycleError:
                        session = ctx.session.with_status(DistributedStatus.FAILED)
                    self._ctx = ctx.with_session(session)
            else:
                if ctx.session.status is DistributedStatus.RUNNING:
                    session = _transition(ctx.session, DistributedStatus.DRAINING)
                    self._ctx = ctx.with_session(session)
                    session = _transition(self._ctx.session, DistributedStatus.COMPLETED)
                    self._ctx = self._ctx.with_session(session)
                elif ctx.session.status is DistributedStatus.READY:
                    # Opened but never ran — complete via drain path from READY? not allowed.
                    # Mark aborted cleanly for unused runtime.
                    try:
                        session = _transition(ctx.session, DistributedStatus.ABORTED)
                    except DistributedLifecycleError:
                        session = ctx.session.with_status(DistributedStatus.ABORTED)
                    self._ctx = ctx.with_session(session)
            ctx.backend.finalize()
        finally:
            self._sync = None
            # keep last context for inspection
            pass

    def _require(self) -> DistributedContext:
        if self._ctx is None:
            raise DistributedError("DistributedRuntime is not open.")
        return self._ctx
