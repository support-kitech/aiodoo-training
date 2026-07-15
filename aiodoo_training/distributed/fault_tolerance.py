"""FaultToleranceCoordinator — RestartPolicy beside frozen ResumePolicy."""

from __future__ import annotations

from dataclasses import dataclass

from aiodoo_training.domain.distributed_health import DistributedHealth
from aiodoo_training.domain.distributed_policies import RestartPolicy
from aiodoo_training.domain.enums import ClusterStatus, RestartFrom


@dataclass(frozen=True, slots=True)
class RestartDecision:
    """Immutable decision — does not validate checkpoints (ResumePolicy does)."""

    should_restart: bool
    restart_from: RestartFrom
    reason: str
    attempts_remaining: int


class FaultToleranceCoordinator:
    """
    Observes DistributedHealth and applies RestartPolicy.

    Never softens ResumePolicy; resume path still validates checkpoints.
    """

    def __init__(self, policy: RestartPolicy | None = None) -> None:
        self._policy = policy or RestartPolicy()
        self._restarts = 0

    @property
    def restart_count(self) -> int:
        return self._restarts

    def on_incident(
        self,
        health: DistributedHealth,
        *,
        current_world_size: int,
        checkpoint_world_size: int | None = None,
        current_mesh_digest: str = "",
        checkpoint_mesh_digest: str | None = None,
    ) -> RestartDecision:
        if health.cluster in {ClusterStatus.HEALTHY, ClusterStatus.UNKNOWN}:
            return RestartDecision(
                should_restart=False,
                restart_from=self._policy.restart_from,
                reason="cluster healthy",
                attempts_remaining=self._policy.max_restarts - self._restarts,
            )
        if self._restarts >= self._policy.max_restarts:
            return RestartDecision(
                should_restart=False,
                restart_from=self._policy.restart_from,
                reason="max_restarts exhausted",
                attempts_remaining=0,
            )
        if (
            self._policy.require_same_world_size
            and checkpoint_world_size is not None
            and checkpoint_world_size != current_world_size
        ):
            return RestartDecision(
                should_restart=False,
                restart_from=self._policy.restart_from,
                reason="world_size mismatch (RestartPolicy)",
                attempts_remaining=self._policy.max_restarts - self._restarts,
            )
        if (
            self._policy.require_same_mesh_digest
            and checkpoint_mesh_digest is not None
            and checkpoint_mesh_digest != current_mesh_digest
        ):
            return RestartDecision(
                should_restart=False,
                restart_from=self._policy.restart_from,
                reason="mesh_digest mismatch (RestartPolicy)",
                attempts_remaining=self._policy.max_restarts - self._restarts,
            )
        self._restarts += 1
        return RestartDecision(
            should_restart=True,
            restart_from=self._policy.restart_from,
            reason=health.message or f"cluster={health.cluster.value}",
            attempts_remaining=self._policy.max_restarts - self._restarts,
        )
