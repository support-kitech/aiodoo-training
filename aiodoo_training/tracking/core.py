"""Phase 6 tracking application: context, lifecycles, coordinator, stores."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from aiodoo_training.domain.artifact_history import (
    ArtifactHistoryEntry,
    ArtifactLineage,
    ArtifactRelationship,
)
from aiodoo_training.domain.cli_profile import CLIProfile
from aiodoo_training.domain.enums import (
    ArtifactRelationKind,
    ExperimentStatus,
    RunState,
    TrackingHealthStatus,
    TrackingSinkStatus,
)
from aiodoo_training.domain.experiment_session import ExperimentHistory, ExperimentSession
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.metric_series import MetricSeries, MetricTimeline
from aiodoo_training.domain.provenance import ExperimentProvenance
from aiodoo_training.domain.run_record import RunMetadata, RunRecord
from aiodoo_training.domain.tracking_policies import (
    TRACKING_PROTOCOL_VERSION,
    LoggingPolicy,
    ReportPolicy,
    RetentionPolicy,
    TrackingCapability,
    TrackingHealth,
    TrackingPolicy,
)
from aiodoo_training.domain.tracking_reports import (
    EvaluationReportSummary,
    ExperimentSummary,
    ExportReport,
    ExportStatistics,
    RunSummary,
    TrainingReport,
)
from aiodoo_training.domain.training import MetricSnapshot
from aiodoo_training.exceptions import TrackingError, TrackingLifecycleError
from aiodoo_training.ports.trainer import ExperimentTracker

_SINK_VALID: dict[TrackingSinkStatus, frozenset[TrackingSinkStatus]] = {
    TrackingSinkStatus.CLOSED: frozenset({TrackingSinkStatus.OPEN}),
    TrackingSinkStatus.OPEN: frozenset(
        {
            TrackingSinkStatus.FLUSHING,
            TrackingSinkStatus.CLOSED,
            TrackingSinkStatus.DEGRADED,
        }
    ),
    TrackingSinkStatus.FLUSHING: frozenset({TrackingSinkStatus.OPEN, TrackingSinkStatus.CLOSED}),
    TrackingSinkStatus.DEGRADED: frozenset({TrackingSinkStatus.OPEN, TrackingSinkStatus.CLOSED}),
}


@dataclass(frozen=True, slots=True)
class TrackingContext:
    """Resolved collaborators for a tracking run (bindable bag)."""

    policy: TrackingPolicy
    capability: TrackingCapability
    health: TrackingHealth
    experiment_session: ExperimentSession
    run_record: RunRecord
    root_dir: Path
    provenance: ExperimentProvenance | None = None
    logging_policy: LoggingPolicy = field(default_factory=LoggingPolicy)
    report_policy: ReportPolicy = field(default_factory=ReportPolicy)
    retention_policy: RetentionPolicy = field(default_factory=RetentionPolicy)
    cli_profile: CLIProfile = field(default_factory=CLIProfile)
    sink_status: TrackingSinkStatus = TrackingSinkStatus.CLOSED
    bind_extra: dict[str, Any] = field(default_factory=dict)

    def with_health(self, health: TrackingHealth) -> TrackingContext:
        return replace(self, health=health)

    def with_run(self, run_record: RunRecord) -> TrackingContext:
        return replace(self, run_record=run_record)

    def with_sink_status(self, status: TrackingSinkStatus) -> TrackingContext:
        return replace(self, sink_status=status)

    def with_provenance(self, provenance: ExperimentProvenance) -> TrackingContext:
        return replace(self, provenance=provenance)


class TrackingLifecycle:
    """Owns allowed tracking sink session transitions (COW)."""

    def transition(self, ctx: TrackingContext, target: TrackingSinkStatus) -> TrackingContext:
        allowed = _SINK_VALID.get(ctx.sink_status, frozenset())
        if target not in allowed:
            raise TrackingLifecycleError(
                f"Cannot transition sink from {ctx.sink_status.value!r} to {target.value!r}."
            )
        return ctx.with_sink_status(target)

    def open(self, ctx: TrackingContext) -> TrackingContext:
        return self.transition(ctx, TrackingSinkStatus.OPEN)

    def flush(self, ctx: TrackingContext) -> TrackingContext:
        return self.transition(ctx, TrackingSinkStatus.FLUSHING)

    def degrade(self, ctx: TrackingContext) -> TrackingContext:
        return self.transition(ctx, TrackingSinkStatus.DEGRADED)

    def close(self, ctx: TrackingContext) -> TrackingContext:
        return self.transition(ctx, TrackingSinkStatus.CLOSED)


class ExperimentLifecycle:
    """Allowed ExperimentStatus transitions."""

    _VALID: dict[ExperimentStatus, frozenset[ExperimentStatus]] = {
        ExperimentStatus.PENDING: frozenset({ExperimentStatus.ACTIVE, ExperimentStatus.FAILED}),
        ExperimentStatus.ACTIVE: frozenset(
            {ExperimentStatus.ARCHIVED, ExperimentStatus.FAILED, ExperimentStatus.ACTIVE}
        ),
        ExperimentStatus.FAILED: frozenset({ExperimentStatus.PENDING}),
        ExperimentStatus.ARCHIVED: frozenset(),
    }

    def transition(
        self, session: ExperimentSession, target: ExperimentStatus, *, message: str | None = None
    ) -> ExperimentSession:
        allowed = self._VALID.get(session.status, frozenset())
        if target not in allowed:
            raise TrackingLifecycleError(
                f"Cannot transition experiment from {session.status.value!r} to {target.value!r}."
            )
        return session.with_status(target, message=message)


class RunLifecycle:
    """Observational RunState transitions (mirrors only)."""

    _VALID: dict[RunState, frozenset[RunState]] = {
        RunState.PENDING: frozenset({RunState.RUNNING, RunState.RESUMED}),
        RunState.RUNNING: frozenset({RunState.COMPLETED, RunState.FAILED, RunState.ABORTED}),
        RunState.RESUMED: frozenset({RunState.RUNNING}),
        RunState.COMPLETED: frozenset({RunState.RESUMED}),
        RunState.FAILED: frozenset({RunState.RESUMED}),
        RunState.ABORTED: frozenset({RunState.RESUMED}),
    }

    def transition(self, record: RunRecord, target: RunState) -> RunRecord:
        allowed = self._VALID.get(record.state, frozenset())
        if target not in allowed:
            raise TrackingLifecycleError(
                f"Cannot transition run from {record.state.value!r} to {target.value!r}."
            )
        return record.with_state(target)


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, sort_keys=True, indent=2, default=str) + "\n"
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(data), sort_keys=True, default=str) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


class MetricStore:
    """Append-only metric persistence under a run directory."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._timeline = MetricTimeline()
        self._series: dict[str, MetricSeries] = {}

    @property
    def timeline(self) -> MetricTimeline:
        return self._timeline.sorted()

    def append(self, snapshot: MetricSnapshot) -> None:
        self._timeline = self._timeline.append(snapshot)
        series = self._series.get(snapshot.name) or MetricSeries(name=snapshot.name)
        self._series[snapshot.name] = series.append(snapshot)
        _append_jsonl(
            self._root / "metrics.jsonl",
            {
                "name": snapshot.name,
                "value": snapshot.value,
                "step": snapshot.step,
                "tags": list(snapshot.tags),
            },
        )

    def record_blob(self, name: str, payload: Mapping[str, object] | object) -> None:
        path = self._root / "history" / "metrics" / f"{name}.json"
        data: dict[str, Any]
        if isinstance(payload, Mapping):
            data = {str(k): v for k, v in payload.items()}
        else:
            from dataclasses import asdict, is_dataclass

            if is_dataclass(payload) and not isinstance(payload, type):
                data = asdict(payload)
            else:
                data = {"repr": repr(payload), "type": type(payload).__name__}
        _write_json(path, data)

    def read_series(self, name: str) -> MetricSeries | None:
        return self._series.get(name)


class ArtifactHistoryStore:
    """Local observational artifact index."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._entries: list[ArtifactHistoryEntry] = []
        self._lineage = ArtifactLineage()

    @property
    def lineage(self) -> ArtifactLineage:
        return self._lineage

    def append(self, entry: ArtifactHistoryEntry) -> None:
        self._entries.append(entry)
        _append_jsonl(
            self._root / "artifacts.jsonl",
            {
                "path": entry.path,
                "role": entry.role,
                "digest": entry.digest,
                "run_id": entry.run_id.value if entry.run_id else None,
            },
        )

    def add_relationship(self, edge: ArtifactRelationship) -> None:
        self._lineage = self._lineage.with_edge(edge)
        _append_jsonl(
            self._root / "history" / "artifacts" / "lineage.jsonl",
            {
                "kind": edge.kind.value,
                "source": edge.source,
                "target": edge.target,
                "run_id": edge.run_id.value if edge.run_id else None,
            },
        )

    def entries(self) -> tuple[ArtifactHistoryEntry, ...]:
        return tuple(self._entries)


class MetadataStore:
    """Experiment catalog + run index documents."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._experiments: dict[str, ExperimentSummary] = {}
        self._runs: dict[str, RunSummary] = {}
        self._load()

    def _load(self) -> None:
        exp_idx = _read_json(self._root / "indexes" / "experiments.json")
        for key, raw in (exp_idx.get("experiments") or {}).items():
            self._experiments[key] = ExperimentSummary(
                experiment_id=ExperimentId(value=raw["experiment_id"]),
                name=raw["name"],
                status=raw["status"],
                run_count=int(raw.get("run_count", 0)),
                latest_run_id=(
                    RunId(value=raw["latest_run_id"]) if raw.get("latest_run_id") else None
                ),
                config_fingerprint=str(raw.get("config_fingerprint", "")),
                tracking_protocol_version=str(
                    raw.get("tracking_protocol_version", TRACKING_PROTOCOL_VERSION)
                ),
            )
        run_idx = _read_json(self._root / "indexes" / "runs.json")
        for key, raw in (run_idx.get("runs") or {}).items():
            self._runs[key] = RunSummary(
                run_id=RunId(value=raw["run_id"]),
                experiment_id=ExperimentId(value=raw["experiment_id"]),
                state=RunState(raw["state"]),
                provenance_digest=str(raw.get("provenance_digest", "")),
                packing_fingerprint=raw.get("packing_fingerprint"),
                curriculum_fingerprint=raw.get("curriculum_fingerprint"),
                tracking_protocol_version=str(
                    raw.get("tracking_protocol_version", TRACKING_PROTOCOL_VERSION)
                ),
            )

    def upsert_experiment(self, summary: ExperimentSummary) -> None:
        self._experiments[summary.experiment_id.value] = summary
        self._flush_experiments()

    def upsert_run(self, summary: RunSummary) -> None:
        self._runs[summary.run_id.value] = summary
        self._flush_runs()

    def _flush_experiments(self) -> None:
        payload = {
            "experiments": {
                k: {
                    "experiment_id": v.experiment_id.value,
                    "name": v.name,
                    "status": v.status,
                    "run_count": v.run_count,
                    "latest_run_id": v.latest_run_id.value if v.latest_run_id else None,
                    "config_fingerprint": v.config_fingerprint,
                    "tracking_protocol_version": v.tracking_protocol_version,
                }
                for k, v in sorted(self._experiments.items())
            }
        }
        _write_json(self._root / "indexes" / "experiments.json", payload)

    def _flush_runs(self) -> None:
        payload = {
            "runs": {
                k: {
                    "run_id": v.run_id.value,
                    "experiment_id": v.experiment_id.value,
                    "state": v.state.value,
                    "provenance_digest": v.provenance_digest,
                    "packing_fingerprint": v.packing_fingerprint,
                    "curriculum_fingerprint": v.curriculum_fingerprint,
                    "tracking_protocol_version": v.tracking_protocol_version,
                }
                for k, v in sorted(self._runs.items())
            }
        }
        _write_json(self._root / "indexes" / "runs.json", payload)

    def list_experiments(self) -> tuple[ExperimentSummary, ...]:
        return tuple(self._experiments[k] for k in sorted(self._experiments))

    def get_experiment(self, experiment_id: ExperimentId) -> ExperimentSummary | None:
        return self._experiments.get(experiment_id.value)

    def list_runs(self) -> tuple[RunSummary, ...]:
        return tuple(self._runs[k] for k in sorted(self._runs))

    def get_run(self, run_id: RunId) -> RunSummary | None:
        return self._runs.get(run_id.value)


class TrackingStore:
    """Filesystem layout root for a tracking tree."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def experiment_dir(self, experiment_id: ExperimentId) -> Path:
        return self.root / "experiments" / experiment_id.value

    def run_dir(self, experiment_id: ExperimentId, run_id: RunId) -> Path:
        return self.experiment_dir(experiment_id) / "runs" / run_id.value


class ExperimentRegistry:
    """In-memory + MetadataStore-backed experiment registry."""

    def __init__(self, metadata: MetadataStore) -> None:
        self._metadata = metadata
        self._sessions: dict[str, ExperimentSession] = {}
        self._histories: dict[str, ExperimentHistory] = {}

    def register(self, session: ExperimentSession) -> None:
        self._sessions[session.experiment_id.value] = session
        self._histories.setdefault(
            session.experiment_id.value,
            ExperimentHistory(experiment_id=session.experiment_id),
        )
        self._metadata.upsert_experiment(
            ExperimentSummary(
                experiment_id=session.experiment_id,
                name=session.name,
                status=session.status.value,
                run_count=session.run_count,
                latest_run_id=session.latest_run_id,
                config_fingerprint=session.config_fingerprint,
                tracking_protocol_version=session.tracking_protocol_version,
            )
        )

    def get(self, experiment_id: ExperimentId) -> ExperimentSession | None:
        return self._sessions.get(experiment_id.value)

    def history(self, experiment_id: ExperimentId) -> ExperimentHistory:
        return self._histories.get(
            experiment_id.value, ExperimentHistory(experiment_id=experiment_id)
        )

    def note_run(self, experiment_id: ExperimentId, run_id: RunId) -> ExperimentSession | None:
        session = self._sessions.get(experiment_id.value)
        if session is None:
            return None
        updated = session.with_run(run_id)
        self._sessions[experiment_id.value] = updated
        hist = self.history(experiment_id).with_run(run_id)
        self._histories[experiment_id.value] = hist
        self.register(updated)
        return updated


class ExperimentCatalog:
    """Query facade over MetadataStore."""

    def __init__(self, metadata: MetadataStore) -> None:
        self._metadata = metadata

    def list(self) -> Sequence[ExperimentSummary]:
        return self._metadata.list_experiments()

    def get(self, experiment_id: ExperimentId) -> ExperimentSummary | None:
        return self._metadata.get_experiment(experiment_id)


class RunIndex:
    """Query facade over run summaries."""

    def __init__(self, metadata: MetadataStore) -> None:
        self._metadata = metadata

    def list(self) -> Sequence[RunSummary]:
        return self._metadata.list_runs()

    def get(self, run_id: RunId) -> RunSummary | None:
        return self._metadata.get_run(run_id)


class TrackingCoordinator:
    """
    Sole Phase 6 observational orchestrator.

    Fans events/DTOs to ExperimentTracker + stores. Never owns training authority.
    """

    def __init__(
        self,
        *,
        tracker: ExperimentTracker,
        context: TrackingContext,
        store: TrackingStore,
        metadata: MetadataStore | None = None,
        registry: ExperimentRegistry | None = None,
        lifecycle: TrackingLifecycle | None = None,
        run_lifecycle: RunLifecycle | None = None,
        experiment_lifecycle: ExperimentLifecycle | None = None,
    ) -> None:
        self._tracker = tracker
        self._ctx = context
        self._store = store
        self._metadata = metadata or MetadataStore(store.root)
        self._registry = registry or ExperimentRegistry(self._metadata)
        self._life = lifecycle or TrackingLifecycle()
        self._run_life = run_lifecycle or RunLifecycle()
        self._exp_life = experiment_lifecycle or ExperimentLifecycle()
        self._run_root = store.run_dir(
            context.experiment_session.experiment_id, context.run_record.run_id
        )
        self._metrics = MetricStore(self._run_root)
        self._artifacts = ArtifactHistoryStore(self._run_root)
        self._run_index = RunIndex(self._metadata)
        if hasattr(tracker, "bind"):
            tracker.bind(context)

    @property
    def context(self) -> TrackingContext:
        return self._ctx

    @property
    def metric_store(self) -> MetricStore:
        return self._metrics

    @property
    def artifact_history(self) -> ArtifactHistoryStore:
        return self._artifacts

    def open(self) -> None:
        try:
            self._ctx = self._life.open(self._ctx)
            self._registry.register(self._ctx.experiment_session)
            self._registry.note_run(
                self._ctx.experiment_session.experiment_id, self._ctx.run_record.run_id
            )
            record = self._run_life.transition(self._ctx.run_record, RunState.RUNNING)
            self._ctx = self._ctx.with_run(record)
            self._persist_run()
            if self._ctx.capability.supports_params:
                params: dict[str, object] = {
                    "experiment_id": self._ctx.experiment_session.experiment_id.value,
                    "run_id": self._ctx.run_record.run_id.value,
                    "backend": self._ctx.policy.backend_key,
                }
                if self._ctx.provenance is not None:
                    params["provenance_digest"] = self._ctx.provenance.digest
                self._safe_track(lambda: self._tracker.log_params(params))
            _write_json(
                self._run_root / "run.json",
                {
                    "run_id": self._ctx.run_record.run_id.value,
                    "experiment_id": self._ctx.run_record.experiment_id.value,
                    "state": self._ctx.run_record.state.value,
                    "tracking_protocol_version": (self._ctx.run_record.tracking_protocol_version),
                },
            )
        except TrackingLifecycleError:
            raise
        except Exception as exc:  # noqa: BLE001 — observational isolation
            self._degrade(str(exc))
            if not self._ctx.policy.nonfatal_sink_errors:
                raise TrackingError(str(exc)) from exc

    def observe_metrics(self, metrics: Sequence[MetricSnapshot]) -> None:
        if not self._ctx.capability.supports_metrics:
            return
        for snapshot in metrics:
            self._metrics.append(snapshot)
        self._safe_track(lambda: self._tracker.log_metrics(tuple(metrics)))

    def observe_statistics_blob(self, name: str, payload: object) -> None:
        self._metrics.record_blob(name, payload)

    def observe_artifact(self, path: Path, *, role: str = "artifact", digest: str = "") -> None:
        entry = ArtifactHistoryEntry(
            path=str(path),
            role=role,
            digest=digest,
            run_id=self._ctx.run_record.run_id,
        )
        self._artifacts.append(entry)
        if self._ctx.capability.supports_lineage:
            self._artifacts.add_relationship(
                ArtifactRelationship(
                    kind=ArtifactRelationKind.PRODUCED_BY,
                    source=self._ctx.run_record.run_id.value,
                    target=str(path),
                    run_id=self._ctx.run_record.run_id,
                )
            )
        if self._ctx.capability.supports_artifacts:
            self._safe_track(lambda: self._tracker.log_artifact(path, name=role))

    def complete(self, state: RunState = RunState.COMPLETED) -> None:
        try:
            record = self._run_life.transition(self._ctx.run_record, state)
            self._ctx = self._ctx.with_run(record)
            self._persist_run()
            self.close()
        except Exception as exc:  # noqa: BLE001
            self._degrade(str(exc))
            if not self._ctx.policy.nonfatal_sink_errors:
                raise TrackingError(str(exc)) from exc

    def close(self) -> None:
        try:
            if self._ctx.sink_status is TrackingSinkStatus.OPEN:
                self._ctx = self._life.flush(self._ctx)
            if self._ctx.sink_status in {
                TrackingSinkStatus.FLUSHING,
                TrackingSinkStatus.OPEN,
                TrackingSinkStatus.DEGRADED,
            }:
                if self._ctx.sink_status is TrackingSinkStatus.FLUSHING:
                    self._ctx = self._life.close(self._ctx)
                elif self._ctx.sink_status is TrackingSinkStatus.DEGRADED:
                    self._ctx = self._life.close(self._ctx)
                elif self._ctx.sink_status is TrackingSinkStatus.OPEN:
                    self._ctx = self._life.close(self._ctx)
            self._safe_track(self._tracker.close)
            self._ctx = self._ctx.with_health(
                TrackingHealth(
                    backend_key=self._ctx.policy.backend_key,
                    status=TrackingHealthStatus.HEALTHY,
                    message="closed",
                    last_success_at=datetime.now(UTC),
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._degrade(str(exc))

    def _persist_run(self) -> None:
        rec = self._ctx.run_record
        self._metadata.upsert_run(
            RunSummary(
                run_id=rec.run_id,
                experiment_id=rec.experiment_id,
                state=rec.state,
                provenance_digest=rec.provenance_digest,
                packing_fingerprint=rec.packing_fingerprint,
                curriculum_fingerprint=rec.curriculum_fingerprint,
                tracking_protocol_version=rec.tracking_protocol_version,
            )
        )

    def _safe_track(self, fn: Any) -> None:
        try:
            fn()
            self._ctx = self._ctx.with_health(
                TrackingHealth(
                    backend_key=self._ctx.policy.backend_key,
                    status=TrackingHealthStatus.HEALTHY,
                    last_success_at=datetime.now(UTC),
                    consecutive_failures=0,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._degrade(str(exc))
            if not self._ctx.policy.nonfatal_sink_errors:
                raise TrackingError(str(exc)) from exc

    def _degrade(self, message: str) -> None:
        failures = self._ctx.health.consecutive_failures + 1
        self._ctx = self._ctx.with_health(
            TrackingHealth(
                backend_key=self._ctx.policy.backend_key,
                status=TrackingHealthStatus.DEGRADED,
                message=message,
                consecutive_failures=failures,
            )
        )
        if self._ctx.sink_status is TrackingSinkStatus.OPEN:
            try:
                self._ctx = self._life.degrade(self._ctx)
            except TrackingLifecycleError:
                pass


def new_experiment_session(
    *,
    experiment_id: ExperimentId,
    name: str,
    config_fingerprint: str = "",
    model_fingerprint: str = "",
    adapter_fingerprint: str = "",
) -> ExperimentSession:
    now = datetime.now(UTC)
    return ExperimentSession(
        session_id=f"exp-{uuid4().hex[:12]}",
        experiment_id=experiment_id,
        name=name,
        created_at=now,
        updated_at=now,
        config_fingerprint=config_fingerprint,
        model_fingerprint=model_fingerprint,
        adapter_fingerprint=adapter_fingerprint,
    )


def new_run_record(
    *,
    experiment_id: ExperimentId,
    run_id: RunId | None = None,
    metadata: RunMetadata | None = None,
) -> RunRecord:
    return RunRecord(
        run_id=run_id or RunId(value=f"run-{uuid4().hex[:12]}"),
        experiment_id=experiment_id,
        started_at=datetime.now(UTC),
        metadata=metadata or RunMetadata(),
    )


# Re-exports used by reports
TrainingReport = TrainingReport
EvaluationReportSummary = EvaluationReportSummary
ExportReport = ExportReport
ExportStatistics = ExportStatistics
