"""Export session lifecycle — owns allowed state transitions (COW)."""

from __future__ import annotations

from aiodoo_training.domain.enums import ExportStatus
from aiodoo_training.domain.export_session import ExportSession
from aiodoo_training.exceptions import ExportLifecycleError

_VALID_TRANSITIONS: dict[ExportStatus, frozenset[ExportStatus]] = {
    ExportStatus.PENDING: frozenset({ExportStatus.VALIDATING, ExportStatus.FAILED}),
    ExportStatus.VALIDATING: frozenset({ExportStatus.PACKAGING, ExportStatus.FAILED}),
    ExportStatus.PACKAGING: frozenset({ExportStatus.PUBLISHED, ExportStatus.FAILED}),
    ExportStatus.PUBLISHED: frozenset(),
    ExportStatus.FAILED: frozenset(),
}


class ExportLifecycle:
    """
    Application owner of export session lifecycle transitions.

    Never mutates domain snapshots in place; always returns a new
    :class:`ExportSession` via copy-on-write helpers.
    """

    def _transition(
        self,
        session: ExportSession,
        target: ExportStatus,
        *,
        message: str | None = None,
    ) -> ExportSession:
        allowed = _VALID_TRANSITIONS.get(session.status, frozenset())
        if target not in allowed:
            raise ExportLifecycleError(
                f"Cannot transition from {session.status.value!r} to {target.value!r}."
            )
        return session.with_status(target, message=message)

    def preflight(self, session: ExportSession, *, message: str | None = None) -> ExportSession:
        """``PENDING`` → ``VALIDATING``."""
        return self._transition(session, ExportStatus.VALIDATING, message=message)

    def begin_packaging(
        self, session: ExportSession, *, message: str | None = None
    ) -> ExportSession:
        """``VALIDATING`` → ``PACKAGING``."""
        return self._transition(session, ExportStatus.PACKAGING, message=message)

    def publish(self, session: ExportSession, *, message: str | None = None) -> ExportSession:
        """``PACKAGING`` → ``PUBLISHED``."""
        return self._transition(session, ExportStatus.PUBLISHED, message=message)

    def fail(self, session: ExportSession, *, message: str | None = None) -> ExportSession:
        """Any active state → ``FAILED`` (via allowed transitions)."""
        if session.status == ExportStatus.PENDING:
            return self._transition(session, ExportStatus.FAILED, message=message)
        if session.status == ExportStatus.VALIDATING:
            return self._transition(session, ExportStatus.FAILED, message=message)
        if session.status == ExportStatus.PACKAGING:
            return self._transition(session, ExportStatus.FAILED, message=message)
        raise ExportLifecycleError(
            f"Cannot fail export session from status {session.status.value!r}."
        )
