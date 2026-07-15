"""Phase 6 tracking policy / capability / health domain DTOs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

from aiodoo_training.domain.enums import TrackerType, TrackingHealthStatus

# Observational layout / serialization version for tracking *history* only.
# Never gates ExperimentId, fingerprints, ResumePolicy, train, eval, export,
# or Artifact Contract.
TRACKING_PROTOCOL_VERSION = "1"


@dataclass(frozen=True, slots=True)
class TrackingCapability:
    """Immutable declared feature flags for a tracking backend."""

    backend_key: str
    supports_metrics: bool = True
    supports_artifacts: bool = True
    supports_params: bool = True
    supports_lineage: bool = False
    supports_live_stream: bool = False
    supports_resume: bool = False
    supports_remote: bool = False
    supports_tags: bool = True

    def supports(self, feature: str) -> bool:
        """Convenience lookup for a declared capability flag.

        Accepts ``\"metrics\"`` or ``\"supports_metrics\"``. Unknown features
        return ``False``. Does not change public behaviour of callers that
        already read boolean fields directly.
        """
        name = feature.strip()
        if not name:
            return False
        if not name.startswith("supports_"):
            name = f"supports_{name}"
        known = {f.name for f in fields(self)}
        if name not in known:
            return False
        return bool(getattr(self, name))


@dataclass(frozen=True, slots=True)
class TrackingHealth:
    """Immutable backend sink health snapshot (never training health)."""

    backend_key: str
    status: TrackingHealthStatus
    message: str | None = None
    last_success_at: datetime | None = None
    consecutive_failures: int = 0


@dataclass(frozen=True, slots=True)
class TrackingPolicy:
    """Resolved tracking policy from TrackingSpec + Phase 6 fragments."""

    backend_key: str = "null"
    tracker_type: TrackerType = TrackerType.NULL
    enabled: bool = True
    experiment_name: str | None = None
    tracking_uri: str | None = None
    root_dir: Path | None = None
    flush_every_n_steps: int = 50
    nonfatal_sink_errors: bool = True
    tags: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class LoggingPolicy:
    """Operational logging policy (separate from ExperimentTracker)."""

    level: str = "INFO"
    sinks: tuple[str, ...] = ("console",)
    jsonl_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ReportPolicy:
    """Which report formats to emit under the tracking root."""

    write_json: bool = True
    write_markdown: bool = False


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Tracking-store retention / rotation (never checkpoint packages)."""

    max_runs_per_experiment: int = 50
    max_metric_files: int = 100
    keep_failed: bool = True
