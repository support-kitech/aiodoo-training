"""Phase 6 tracking / logging / retention / CLI configuration fragments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aiodoo_training.domain.cli_profile import CLIProfile, resolve_cli_profile
from aiodoo_training.domain.enums import CLIProfileName, TrackerType
from aiodoo_training.domain.tracking_policies import (
    LoggingPolicy,
    ReportPolicy,
    RetentionPolicy,
    TrackingPolicy,
)
from aiodoo_training.exceptions import ConfigError


class TrackingFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "null"
    enabled: bool = True
    experiment_name: str | None = None
    tracking_uri: str | None = None
    root_dir: str | None = None
    flush_every_n_steps: int = Field(default=50, ge=1)
    nonfatal_sink_errors: bool = True
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("backend")
    @classmethod
    def _backend_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tracking.backend must be non-empty")
        return value


class LoggingFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    sinks: list[str] = Field(default_factory=lambda: ["console"])
    jsonl_path: str | None = None


class ReportsFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    write_json: bool = True
    write_markdown: bool = False


class RetentionFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_runs_per_experiment: int = Field(default=50, ge=1)
    max_metric_files: int = Field(default=100, ge=1)
    keep_failed: bool = True


class CLIFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: Literal["default", "minimal", "verbose", "json", "ci"] = "default"
    progress: bool | None = None
    color: Literal["auto", "always", "never"] | None = None
    default_output: Literal["text", "json"] | None = None


def parse_tracking_config(raw: dict[str, Any] | None) -> TrackingFragment:
    try:
        return TrackingFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid tracking config: {exc}") from exc


def parse_logging_config(raw: dict[str, Any] | None) -> LoggingFragment:
    try:
        return LoggingFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid logging config: {exc}") from exc


def parse_reports_config(raw: dict[str, Any] | None) -> ReportsFragment:
    try:
        return ReportsFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid reports config: {exc}") from exc


def parse_retention_config(raw: dict[str, Any] | None) -> RetentionFragment:
    try:
        return RetentionFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid retention config: {exc}") from exc


def parse_cli_config(raw: dict[str, Any] | None) -> CLIFragment:
    try:
        return CLIFragment.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid cli config: {exc}") from exc


def to_tracking_policy(fragment: TrackingFragment) -> TrackingPolicy:
    try:
        tracker_type = TrackerType(fragment.backend)
    except ValueError:
        tracker_type = TrackerType.NULL
    return TrackingPolicy(
        backend_key=fragment.backend,
        tracker_type=tracker_type,
        enabled=fragment.enabled,
        experiment_name=fragment.experiment_name,
        tracking_uri=fragment.tracking_uri,
        root_dir=Path(fragment.root_dir) if fragment.root_dir else None,
        flush_every_n_steps=fragment.flush_every_n_steps,
        nonfatal_sink_errors=fragment.nonfatal_sink_errors,
        tags=tuple(sorted((str(k), str(v)) for k, v in fragment.tags.items())),
    )


def to_logging_policy(fragment: LoggingFragment) -> LoggingPolicy:
    return LoggingPolicy(
        level=fragment.level,
        sinks=tuple(fragment.sinks),
        jsonl_path=Path(fragment.jsonl_path) if fragment.jsonl_path else None,
    )


def to_report_policy(fragment: ReportsFragment) -> ReportPolicy:
    return ReportPolicy(
        write_json=fragment.write_json, write_markdown=fragment.write_markdown
    )


def to_retention_policy(fragment: RetentionFragment) -> RetentionPolicy:
    return RetentionPolicy(
        max_runs_per_experiment=fragment.max_runs_per_experiment,
        max_metric_files=fragment.max_metric_files,
        keep_failed=fragment.keep_failed,
    )


def to_cli_profile(fragment: CLIFragment) -> CLIProfile:
    base = resolve_cli_profile(CLIProfileName(fragment.profile))
    return CLIProfile(
        name=base.name,
        progress=base.progress if fragment.progress is None else fragment.progress,
        color=base.color if fragment.color is None else fragment.color,
        output=base.output if fragment.default_output is None else fragment.default_output,
        verbosity=base.verbosity,
        confirm_destructive=base.confirm_destructive,
    )
