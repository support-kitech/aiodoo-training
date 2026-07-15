"""Null + filesystem + MLflow ExperimentTracker backends (infra quarantine for SDKs)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aiodoo_training.domain.enums import TrackingHealthStatus
from aiodoo_training.domain.tracking_policies import TrackingCapability, TrackingHealth
from aiodoo_training.domain.training import MetricSnapshot
from aiodoo_training.ports.trainer import ExperimentTracker
from aiodoo_training.registries import tracker_registry


class NullTracker(ExperimentTracker):
    """CI default — no I/O."""

    BACKEND_KEY = "null"
    CAPABILITY = TrackingCapability(
        backend_key="null",
        supports_metrics=True,
        supports_artifacts=True,
        supports_params=True,
        supports_lineage=False,
        supports_live_stream=False,
        supports_resume=False,
        supports_remote=False,
    )

    def __init__(self, context: Any | None = None) -> None:
        self._context = context
        self._params: dict[str, object] = {}
        self._metrics: list[MetricSnapshot] = []
        self._artifacts: list[tuple[Path, str | None]] = []
        self._closed = False

    def bind(self, context: Any) -> NullTracker:
        self._context = context
        return self

    def log_params(self, params: dict[str, object]) -> None:
        self._params.update(params)

    def log_metrics(self, metrics: Sequence[MetricSnapshot]) -> None:
        self._metrics.extend(metrics)

    def log_artifact(self, path: Path, name: str | None = None) -> None:
        self._artifacts.append((path, name))

    def close(self) -> None:
        self._closed = True

    def health(self) -> TrackingHealth:
        return TrackingHealth(backend_key="null", status=TrackingHealthStatus.HEALTHY)


class FilesystemTracker(ExperimentTracker):
    """Local JSONL tracker under TrackingContext.root_dir / run path."""

    BACKEND_KEY = "local_jsonl"
    CAPABILITY = TrackingCapability(
        backend_key="local_jsonl",
        supports_metrics=True,
        supports_artifacts=True,
        supports_params=True,
        supports_lineage=True,
        supports_live_stream=False,
        supports_resume=True,
        supports_remote=False,
    )

    def __init__(self, context: Any | None = None) -> None:
        self._context = context
        self._run_dir: Path | None = None

    def bind(self, context: Any) -> FilesystemTracker:
        self._context = context
        root = getattr(context, "root_dir", None)
        run = getattr(context, "run_record", None)
        exp = getattr(context, "experiment_session", None)
        if root is not None and run is not None and exp is not None:
            self._run_dir = (
                Path(root)
                / "experiments"
                / exp.experiment_id.value
                / "runs"
                / run.run_id.value
            )
            self._run_dir.mkdir(parents=True, exist_ok=True)
        return self

    def _append(self, filename: str, payload: dict[str, object]) -> None:
        if self._run_dir is None:
            return
        path = self._run_dir / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def log_params(self, params: dict[str, object]) -> None:
        self._append("params.jsonl", {"params": params})

    def log_metrics(self, metrics: Sequence[MetricSnapshot]) -> None:
        for snapshot in metrics:
            self._append(
                "metrics.jsonl",
                {
                    "name": snapshot.name,
                    "value": snapshot.value,
                    "step": snapshot.step,
                    "tags": list(snapshot.tags),
                },
            )

    def log_artifact(self, path: Path, name: str | None = None) -> None:
        self._append("artifacts.jsonl", {"path": str(path), "name": name})

    def close(self) -> None:
        return

    def health(self) -> TrackingHealth:
        status = TrackingHealthStatus.HEALTHY
        message = None
        if self._run_dir is None:
            status = TrackingHealthStatus.OFFLINE
            message = "run directory not bound"
        return TrackingHealth(backend_key="local_jsonl", status=status, message=message)


class MLflowTracker(ExperimentTracker):
    """
    Optional MLflow adapter.

    Soft-imports mlflow inside methods; if unavailable, operations no-op and
    health reports OFFLINE — train path remains unaffected under nonfatal policy.
    """

    BACKEND_KEY = "mlflow"
    CAPABILITY = TrackingCapability(
        backend_key="mlflow",
        supports_metrics=True,
        supports_artifacts=True,
        supports_params=True,
        supports_lineage=True,
        supports_live_stream=False,
        supports_resume=True,
        supports_remote=True,
    )

    def __init__(self, context: Any | None = None) -> None:
        self._context = context
        self._active = False
        self._client: Any = None

    def bind(self, context: Any) -> MLflowTracker:
        self._context = context
        return self

    def _ensure(self) -> bool:
        if self._client is not None:
            return True
        try:
            import mlflow  # type: ignore[import-not-found]

            uri = None
            if self._context is not None:
                policy = getattr(self._context, "policy", None)
                uri = getattr(policy, "tracking_uri", None) if policy else None
            if uri:
                mlflow.set_tracking_uri(uri)
            self._client = mlflow
            self._active = True
            return True
        except Exception:  # noqa: BLE001
            self._active = False
            self._client = None
            return False

    def log_params(self, params: dict[str, object]) -> None:
        if not self._ensure():
            return
        assert self._client is not None
        self._client.log_params({str(k): str(v) for k, v in params.items()})

    def log_metrics(self, metrics: Sequence[MetricSnapshot]) -> None:
        if not self._ensure():
            return
        assert self._client is not None
        for snapshot in metrics:
            self._client.log_metric(snapshot.name, snapshot.value, step=snapshot.step)

    def log_artifact(self, path: Path, name: str | None = None) -> None:
        if not self._ensure():
            return
        assert self._client is not None
        self._client.log_artifact(str(path), artifact_path=name)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.end_run()
            except Exception:  # noqa: BLE001
                pass

    def health(self) -> TrackingHealth:
        if self._ensure():
            return TrackingHealth(backend_key="mlflow", status=TrackingHealthStatus.HEALTHY)
        return TrackingHealth(
            backend_key="mlflow",
            status=TrackingHealthStatus.OFFLINE,
            message="mlflow not available",
        )


def capability_for(key: str) -> TrackingCapability:
    mapping = {
        "null": NullTracker.CAPABILITY,
        "local_jsonl": FilesystemTracker.CAPABILITY,
        "filesystem": FilesystemTracker.CAPABILITY,
        "mlflow": MLflowTracker.CAPABILITY,
    }
    return mapping.get(key, TrackingCapability(backend_key=key))


def probe_health(tracker: ExperimentTracker) -> TrackingHealth:
    if hasattr(tracker, "health"):
        return tracker.health()  # type: ignore[no-any-return, attr-defined]
    key = getattr(tracker, "BACKEND_KEY", "unknown")
    return TrackingHealth(backend_key=str(key), status=TrackingHealthStatus.HEALTHY)


def register_default_trackers(*, overwrite: bool = False) -> None:
    pairs = (
        ("null", NullTracker),
        ("local_jsonl", FilesystemTracker),
        ("filesystem", FilesystemTracker),
        ("mlflow", MLflowTracker),
    )
    for key, cls in pairs:
        if not tracker_registry.exists(key) or overwrite:
            tracker_registry.register(key, cls, overwrite=overwrite)
