"""CLI command registry, context, and polished commands."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from aiodoo_training import __version__
from aiodoo_training.bootstrap import bootstrap_phase6
from aiodoo_training.config import ConfigSystem
from aiodoo_training.config.tracking_config import (
    parse_cli_config,
    parse_tracking_config,
    to_cli_profile,
    to_tracking_policy,
)
from aiodoo_training.determinism import FingerprintService
from aiodoo_training.domain.cli_profile import CLIProfile, resolve_cli_profile
from aiodoo_training.domain.enums import CLIProfileName
from aiodoo_training.domain.identifiers import ExperimentId
from aiodoo_training.factories.factories import TrackerFactory
from aiodoo_training.infrastructure.tracking import capability_for, probe_health
from aiodoo_training.tracking.core import MetadataStore, TrackingStore


@dataclass
class CommandContext:
    profile: CLIProfile = field(default_factory=CLIProfile)
    dry_run: bool = False
    verbose: bool = False
    json_output: bool = False
    config_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def with_profile(self, profile: CLIProfile) -> CommandContext:
        return replace(self, profile=profile)


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, Callable[[CommandContext], int]] = {}

    def register(self, name: str, fn: Callable[[CommandContext], int]) -> None:
        self._commands[name] = fn

    def dispatch(self, name: str, ctx: CommandContext) -> int:
        if name not in self._commands:
            print(f"Unknown command: {name}")
            return 2
        return self._commands[name](ctx)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._commands))


class CommandBuilder:
    def __init__(self) -> None:
        self._profile_name = "default"
        self._dry_run = False
        self._verbose = False
        self._json = False

    def with_profile(self, name: str) -> CommandBuilder:
        self._profile_name = name
        return self

    def with_dry_run(self, value: bool = True) -> CommandBuilder:
        self._dry_run = value
        return self

    def with_verbose(self, value: bool = True) -> CommandBuilder:
        self._verbose = value
        return self

    def with_json(self, value: bool = True) -> CommandBuilder:
        self._json = value
        return self

    def build(self, *, config_path: Path | None = None) -> CommandContext:
        if os.environ.get("CI") and self._profile_name == "default":
            profile = resolve_cli_profile(CLIProfileName.CI)
        else:
            profile = resolve_cli_profile(self._profile_name)
        if self._verbose:
            profile = CLIProfile(
                name=profile.name,
                progress=profile.progress,
                color=profile.color,
                output=profile.output,
                verbosity=2,
                confirm_destructive=profile.confirm_destructive,
            )
        if self._json:
            profile = CLIProfile(
                name=profile.name,
                progress=False,
                color="never",
                output="json",
                verbosity=profile.verbosity,
                confirm_destructive=profile.confirm_destructive,
            )
        return CommandContext(
            profile=profile,
            dry_run=self._dry_run,
            verbose=self._verbose or profile.verbosity >= 2,
            json_output=profile.output == "json" or self._json,
            config_path=config_path,
        )


def _emit(ctx: CommandContext, payload: dict[str, Any] | str) -> None:
    if ctx.json_output and isinstance(payload, dict):
        print(json.dumps(payload, sort_keys=True, default=str))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
    else:
        print(payload)


def cmd_validate_config(config_path: Path) -> int:
    system = ConfigSystem()
    model, experiment_id, _resolved = system.load_experiment(config_path)
    print(f"Config OK: {model.name}")
    print(f"schema_version: {model.schema_version}")
    print(f"config_experiment_id: {experiment_id.value}")
    return 0


def cmd_fingerprint(config_path: Path) -> int:
    system = ConfigSystem()
    _model, config_experiment_id, resolved = system.load_experiment(config_path)
    composed = system.composer.compose(config_path)
    service = FingerprintService(system.hasher)
    fingerprint = service.experiment_fingerprint(
        composed,
        package_version=__version__,
    )
    print(f"config_experiment_id: {config_experiment_id.value}")
    print(f"experiment_id: {fingerprint.experiment_id.value}")
    print(f"digest: {fingerprint.digest}")
    print(f"config_digest: {fingerprint.config.digest}")
    print(f"version_digest: {fingerprint.versions.digest}")
    print(f"package_digest: {fingerprint.packages.digest}")
    print(f"resolved_keys: {len(resolved)}")
    return 0


def cmd_doctor() -> int:
    bootstrap_phase6(overwrite=True)
    print(f"aiodoo-training: {__version__}")
    import platform
    import sys

    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print("execution: repository-root scripts (not packaged)")
    print("phase: 6 (tracking + CLI)")
    for key in ("null", "local_jsonl", "mlflow"):
        cap = capability_for(key)
        tracker = TrackerFactory().create(key)
        health = probe_health(tracker)
        print(
            f"tracker[{key}]: health={health.status.value} "
            f"metrics={cap.supports_metrics} artifacts={cap.supports_artifacts} "
            f"remote={cap.supports_remote}"
        )
        if hasattr(tracker, "close"):
            tracker.close()
    return 0


def cmd_train(_config_path: Path) -> int:
    raise NotImplementedError(
        "Full train CLI wiring uses Pipeline; use tests/harness or existing train path."
    )


def cmd_resume(_checkpoint_path: Path) -> int:
    raise NotImplementedError("resume is wired through pipeline restore stage.")


def cmd_evaluate(_config_path: Path) -> int:
    raise NotImplementedError("evaluate uses EvaluationEngine; CLI polish lists catalogs.")


def cmd_merge(_adapter_path: Path) -> int:
    raise NotImplementedError("merge is not implemented in Phase 6 polish.")


def cmd_export(_config_path: Path) -> int:
    raise NotImplementedError("export uses ExportManager; CLI polish lists catalogs.")


def cmd_experiments_list(tracking_root: Path) -> int:
    bootstrap_phase6(overwrite=True)
    store = TrackingStore(tracking_root)
    meta = MetadataStore(store.root)
    for item in meta.list_experiments():
        print(
            f"{item.experiment_id.value}\t{item.name}\t{item.status}\truns={item.run_count}"
        )
    return 0


def cmd_runs_list(tracking_root: Path) -> int:
    bootstrap_phase6(overwrite=True)
    store = TrackingStore(tracking_root)
    meta = MetadataStore(store.root)
    for item in meta.list_runs():
        print(
            f"{item.run_id.value}\t{item.experiment_id.value}\t{item.state.value}\t"
            f"{item.provenance_digest[:12]}"
        )
    return 0


def build_default_registry() -> CommandRegistry:
    registry = CommandRegistry()

    def validate(ctx: CommandContext) -> int:
        if ctx.config_path is None:
            print("Missing --config")
            return 2
        return cmd_validate_config(ctx.config_path)

    def fingerprint(ctx: CommandContext) -> int:
        if ctx.config_path is None:
            print("Missing --config")
            return 2
        return cmd_fingerprint(ctx.config_path)

    def doctor(ctx: CommandContext) -> int:
        _ = ctx
        return cmd_doctor()

    def experiments(ctx: CommandContext) -> int:
        root = Path(ctx.extra.get("tracking_root", "artifacts/tracking"))
        return cmd_experiments_list(root)

    def runs(ctx: CommandContext) -> int:
        root = Path(ctx.extra.get("tracking_root", "artifacts/tracking"))
        return cmd_runs_list(root)

    registry.register("validate_config", validate)
    registry.register("fingerprint", fingerprint)
    registry.register("doctor", doctor)
    registry.register("experiments", experiments)
    registry.register("runs", runs)
    return registry


def resolve_profile_from_raw(raw: dict[str, Any] | None) -> CLIProfile:
    return to_cli_profile(parse_cli_config(raw))


def resolve_tracking_from_raw(raw: dict[str, Any] | None) -> object:
    return to_tracking_policy(parse_tracking_config(raw))


# Keep ExperimentId import used for type docs / future show commands
_ = ExperimentId
