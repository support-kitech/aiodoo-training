"""Training session lifecycle — owns allowed state transitions (COW)."""

from __future__ import annotations

from aiodoo_training.domain.enums import TrainingStatus
from aiodoo_training.domain.training_session import TrainingSession
from aiodoo_training.exceptions import TrainingLifecycleError

_VALID_TRANSITIONS: dict[TrainingStatus, frozenset[TrainingStatus]] = {
    TrainingStatus.PENDING: frozenset({TrainingStatus.RUNNING, TrainingStatus.CANCELLED}),
    TrainingStatus.RUNNING: frozenset(
        {
            TrainingStatus.PAUSED,
            TrainingStatus.COMPLETED,
            TrainingStatus.FAILED,
            TrainingStatus.CANCELLED,
        }
    ),
    TrainingStatus.PAUSED: frozenset(
        {TrainingStatus.RUNNING, TrainingStatus.FAILED, TrainingStatus.CANCELLED}
    ),
    TrainingStatus.COMPLETED: frozenset(),
    TrainingStatus.FAILED: frozenset(),
    TrainingStatus.CANCELLED: frozenset(),
}


class TrainingLifecycle:
    """
    Application owner of training session lifecycle transitions.

    Never mutates domain snapshots in place; always returns a new
    :class:`TrainingSession` via copy-on-write helpers.
    """

    def _transition(
        self,
        session: TrainingSession,
        target: TrainingStatus,
        *,
        message: str | None = None,
    ) -> TrainingSession:
        allowed = _VALID_TRANSITIONS.get(session.status, frozenset())
        if target not in allowed:
            raise TrainingLifecycleError(
                f"Cannot transition from {session.status.value!r} to {target.value!r}."
            )
        return session.with_status(target, message=message)

    def start(self, session: TrainingSession, *, message: str | None = None) -> TrainingSession:
        """``PENDING`` → ``RUNNING`` (fresh train or validated resume)."""
        return self._transition(session, TrainingStatus.RUNNING, message=message)

    def pause(self, session: TrainingSession, *, message: str | None = None) -> TrainingSession:
        """``RUNNING`` → ``PAUSED``."""
        return self._transition(session, TrainingStatus.PAUSED, message=message)

    def resume_running(
        self, session: TrainingSession, *, message: str | None = None
    ) -> TrainingSession:
        """``PAUSED`` → ``RUNNING``."""
        return self._transition(session, TrainingStatus.RUNNING, message=message)

    def complete(self, session: TrainingSession, *, message: str | None = None) -> TrainingSession:
        """``RUNNING`` → ``COMPLETED``."""
        return self._transition(session, TrainingStatus.COMPLETED, message=message)

    def fail(self, session: TrainingSession, *, message: str | None = None) -> TrainingSession:
        """``RUNNING`` or ``PAUSED`` → ``FAILED``."""
        return self._transition(session, TrainingStatus.FAILED, message=message)

    def cancel(self, session: TrainingSession, *, message: str | None = None) -> TrainingSession:
        """``PENDING``, ``RUNNING``, or ``PAUSED`` → ``CANCELLED``."""
        return self._transition(session, TrainingStatus.CANCELLED, message=message)
