"""Packing session lifecycle — owns allowed state transitions (COW)."""

from __future__ import annotations

from aiodoo_training.domain.enums import PackingStatus
from aiodoo_training.domain.packing_session import PackingSession
from aiodoo_training.exceptions import PackingLifecycleError

_VALID: dict[PackingStatus, frozenset[PackingStatus]] = {
    PackingStatus.PENDING: frozenset({PackingStatus.PLANNING, PackingStatus.SKIPPED}),
    PackingStatus.PLANNING: frozenset({PackingStatus.READY, PackingStatus.FAILED}),
    PackingStatus.FAILED: frozenset({PackingStatus.PENDING}),
    PackingStatus.READY: frozenset(),
    PackingStatus.SKIPPED: frozenset(),
}


class PackingLifecycle:
    """Application owner of packing session lifecycle transitions."""

    def _transition(
        self,
        session: PackingSession,
        target: PackingStatus,
        *,
        message: str | None = None,
    ) -> PackingSession:
        allowed = _VALID.get(session.status, frozenset())
        if target not in allowed:
            raise PackingLifecycleError(
                f"Cannot transition from {session.status.value!r} to {target.value!r}."
            )
        return session.with_status(target, message=message)

    def begin(self, session: PackingSession, *, message: str | None = None) -> PackingSession:
        return self._transition(session, PackingStatus.PLANNING, message=message)

    def skip(self, session: PackingSession, *, message: str | None = None) -> PackingSession:
        return self._transition(session, PackingStatus.SKIPPED, message=message)

    def ready(self, session: PackingSession, *, message: str | None = None) -> PackingSession:
        return self._transition(session, PackingStatus.READY, message=message)

    def fail(self, session: PackingSession, *, message: str | None = None) -> PackingSession:
        return self._transition(session, PackingStatus.FAILED, message=message)

    def fresh(self, session: PackingSession, *, message: str | None = None) -> PackingSession:
        return self._transition(session, PackingStatus.PENDING, message=message)
