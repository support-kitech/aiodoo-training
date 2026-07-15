"""Evaluation session lifecycle — owns allowed state transitions (COW)."""

from __future__ import annotations

from aiodoo_training.domain.enums import EvaluationStatus
from aiodoo_training.domain.evaluation_session import EvaluationSession
from aiodoo_training.exceptions import EvaluationLifecycleError

_VALID_TRANSITIONS: dict[EvaluationStatus, frozenset[EvaluationStatus]] = {
    EvaluationStatus.PENDING: frozenset(
        {EvaluationStatus.RUNNING, EvaluationStatus.SKIPPED, EvaluationStatus.CANCELLED}
    ),
    EvaluationStatus.RUNNING: frozenset(
        {EvaluationStatus.COMPLETED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED}
    ),
    EvaluationStatus.FAILED: frozenset({EvaluationStatus.PENDING}),
    EvaluationStatus.COMPLETED: frozenset(),
    EvaluationStatus.CANCELLED: frozenset(),
    EvaluationStatus.SKIPPED: frozenset(),
}


class EvaluationLifecycle:
    """
    Application owner of evaluation session lifecycle transitions.

    Never mutates domain snapshots in place; always returns a new
    :class:`EvaluationSession` via copy-on-write helpers.
    """

    def _transition(
        self,
        session: EvaluationSession,
        target: EvaluationStatus,
        *,
        message: str | None = None,
    ) -> EvaluationSession:
        allowed = _VALID_TRANSITIONS.get(session.status, frozenset())
        if target not in allowed:
            raise EvaluationLifecycleError(
                f"Cannot transition from {session.status.value!r} to {target.value!r}."
            )
        return session.with_status(target, message=message)

    def start(self, session: EvaluationSession, *, message: str | None = None) -> EvaluationSession:
        """``PENDING`` → ``RUNNING``."""
        return self._transition(session, EvaluationStatus.RUNNING, message=message)

    def skip(self, session: EvaluationSession, *, message: str | None = None) -> EvaluationSession:
        """``PENDING`` → ``SKIPPED`` (disabled or no datasets)."""
        return self._transition(session, EvaluationStatus.SKIPPED, message=message)

    def complete(
        self, session: EvaluationSession, *, message: str | None = None
    ) -> EvaluationSession:
        """``RUNNING`` → ``COMPLETED``."""
        return self._transition(session, EvaluationStatus.COMPLETED, message=message)

    def fail(self, session: EvaluationSession, *, message: str | None = None) -> EvaluationSession:
        """``RUNNING`` → ``FAILED``."""
        return self._transition(session, EvaluationStatus.FAILED, message=message)

    def cancel(
        self, session: EvaluationSession, *, message: str | None = None
    ) -> EvaluationSession:
        """``PENDING`` or ``RUNNING`` → ``CANCELLED``."""
        return self._transition(session, EvaluationStatus.CANCELLED, message=message)

    def fresh_session(
        self, session: EvaluationSession, *, message: str | None = None
    ) -> EvaluationSession:
        """``FAILED`` → ``PENDING`` for recovery."""
        return self._transition(session, EvaluationStatus.PENDING, message=message)
