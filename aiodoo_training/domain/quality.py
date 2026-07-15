"""Quality gate report domain objects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class QualityFailure:
    """A single quality gate failure or warning."""

    metric_key: str
    message: str
    severity: str = "error"
    observed: float | None = None
    expected: str | None = None


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Outcome of running AcceptancePolicy against an EvaluationReport."""

    passed: bool
    failures: tuple[QualityFailure, ...] = ()
    warnings: tuple[QualityFailure, ...] = ()
    report_refs: tuple[str, ...] = ()
    created_at: datetime | None = None
    details: str | None = None
