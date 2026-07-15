"""Quality gates — run AcceptancePolicy against EvaluationReport."""

from __future__ import annotations

from datetime import UTC, datetime

from aiodoo_training.domain.artifacts import EvaluationReport
from aiodoo_training.domain.enums import ComparisonOp
from aiodoo_training.domain.evaluation_policies import (
    AcceptancePolicy,
    QualityThreshold,
    ThresholdCombine,
    ThresholdSeverity,
)
from aiodoo_training.domain.quality import QualityFailure, QualityReport
from aiodoo_training.domain.training import MetricSnapshot


def _metric_value(report: EvaluationReport, key: str) -> float | None:
    for snapshot in report.metrics:
        if snapshot.name == key:
            return snapshot.value
    return None


def _compare(op: ComparisonOp, observed: float, expected: float) -> bool:
    if op == ComparisonOp.GE:
        return observed >= expected
    if op == ComparisonOp.LE:
        return observed <= expected
    if op == ComparisonOp.EQ:
        return observed == expected
    if op == ComparisonOp.LT:
        return observed < expected
    if op == ComparisonOp.GT:
        return observed > expected
    return False


class ModelValidator:
    """Application service: runs AcceptancePolicy against EvaluationReport."""

    def validate(
        self,
        report: EvaluationReport,
        policy: AcceptancePolicy,
    ) -> QualityReport:
        return QualityGate().validate(report, policy)


class QualityGate:
    """Execute AcceptancePolicy thresholds → QualityReport."""

    def validate(
        self,
        report: EvaluationReport,
        policy: AcceptancePolicy,
    ) -> QualityReport:
        failures: list[QualityFailure] = []
        warnings: list[QualityFailure] = []
        passed_checks: list[bool] = []

        for threshold in policy.thresholds:
            ok, failure = self._check_threshold(report, threshold)
            passed_checks.append(ok)
            if failure is None:
                continue
            if failure.severity == ThresholdSeverity.WARN.value:
                warnings.append(failure)
            else:
                failures.append(failure)

        if not policy.thresholds:
            passed = True
        elif policy.combine == ThresholdCombine.ALL:
            passed = all(passed_checks) if passed_checks else True
        else:
            passed = any(passed_checks) if passed_checks else True

        return QualityReport(
            passed=passed and not failures,
            failures=tuple(failures),
            warnings=tuple(warnings),
            created_at=datetime.now(UTC),
        )

    def _check_threshold(
        self,
        report: EvaluationReport,
        threshold: QualityThreshold,
    ) -> tuple[bool, QualityFailure | None]:
        observed = _metric_value(report, threshold.metric_key)
        if observed is None:
            failure = QualityFailure(
                metric_key=threshold.metric_key,
                message=f"Metric {threshold.metric_key!r} not found in evaluation report.",
                severity=threshold.severity.value,
                expected=f"{threshold.op.value} {threshold.value}",
            )
            return False, failure

        ok = _compare(threshold.op, observed, threshold.value)
        if ok:
            return True, None

        failure = QualityFailure(
            metric_key=threshold.metric_key,
            message=(
                f"{threshold.metric_key}={observed} failed "
                f"{threshold.op.value} {threshold.value}"
            ),
            severity=threshold.severity.value,
            observed=observed,
            expected=f"{threshold.op.value} {threshold.value}",
        )
        return False, failure


def metrics_from_report(report: EvaluationReport) -> dict[str, float]:
    """Extract metric name → value map from report snapshots."""
    out: dict[str, float] = {}
    for snapshot in report.metrics:
        out[snapshot.name] = snapshot.value
    return out


def final_metrics(report: EvaluationReport) -> tuple[MetricSnapshot, ...]:
    """Return report metrics (evaluation backends emit final aggregates)."""
    return report.metrics
