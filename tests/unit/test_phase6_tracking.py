"""Phase 6 unit / golden / determinism tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase6
from aiodoo_training.builders.tracking_builders import TrackingBuilder
from aiodoo_training.cli.commands import CommandBuilder, build_default_registry, cmd_doctor
from aiodoo_training.config.tracking_config import (
    parse_cli_config,
    parse_tracking_config,
    to_cli_profile,
    to_tracking_policy,
)
from aiodoo_training.domain.curriculum_session import CurriculumStatistics
from aiodoo_training.domain.enums import (
    CLIProfileName,
    RunState,
    TrackingHealthStatus,
)
from aiodoo_training.domain.identifiers import ExperimentId
from aiodoo_training.domain.packing_session import PackingStatistics
from aiodoo_training.domain.training import MetricSnapshot
from aiodoo_training.factories import TrackerFactory
from aiodoo_training.infrastructure.tracking import capability_for, probe_health
from aiodoo_training.tracking import (
    TrackingCoordinator,
    TrackingStore,
    build_provenance,
)


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase6(overwrite=True)


def test_tracker_registry_keys() -> None:
    for key in ("null", "local_jsonl", "mlflow"):
        assert TrackerFactory().create(key) is not None


def test_null_tracker_records_in_memory() -> None:
    tracker = TrackerFactory().create("null")
    tracker.log_params({"a": 1})
    tracker.log_metrics([MetricSnapshot(name="loss", value=0.5, step=1)])
    tracker.close()
    assert probe_health(tracker).status is TrackingHealthStatus.HEALTHY


def test_filesystem_tracker_and_coordinator(tmp_path: Path) -> None:
    root = tmp_path / "tracking"
    policy = to_tracking_policy(
        parse_tracking_config({"backend": "local_jsonl", "enabled": True, "root_dir": str(root)})
    )
    builder = (
        TrackingBuilder()
        .with_policy(policy)
        .with_root(root)
        .with_identity(
            experiment_id=ExperimentId(value="exp1"),
            name="demo",
            config_fingerprint="cfg",
        )
    )
    ctx = builder.build_context()
    tracker = TrackerFactory().create("local_jsonl")
    coord = TrackingCoordinator(tracker=tracker, context=ctx, store=TrackingStore(root))
    coord.open()
    coord.observe_metrics([MetricSnapshot(name="loss", value=1.0, step=0)])
    coord.observe_statistics_blob(
        "packing_statistics",
        PackingStatistics(
            packing_fingerprint="p",
            backend_key="none",
            examples_input=1,
            examples_packed=1,
            sequences_emitted=1,
            tokens_content=1,
            tokens_padded=0,
            pad_ratio=0.0,
            mean_examples_per_sequence=1.0,
            max_sequence_length=8,
        ),
    )
    coord.observe_statistics_blob(
        "curriculum_statistics",
        CurriculumStatistics(
            curriculum_fingerprint="c",
            backend_key="none",
            stage_count=1,
            examples_total=1,
            examples_per_stage=(1,),
        ),
    )
    coord.complete(RunState.COMPLETED)
    run_dir = root / "experiments" / "exp1" / "runs" / ctx.run_record.run_id.value
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "history" / "metrics" / "packing_statistics.json").is_file()


def test_provenance_digest_stable() -> None:
    a = build_provenance(config_fingerprint="x", model_fingerprint="m")
    b = build_provenance(config_fingerprint="x", model_fingerprint="m")
    assert a.digest == b.digest


def test_capability_table() -> None:
    assert capability_for("null").supports_remote is False
    assert capability_for("mlflow").supports_remote is True
    assert capability_for("local_jsonl").supports_lineage is True


def test_capability_supports_helper() -> None:
    cap = capability_for("local_jsonl")
    assert cap.supports("metrics") is True
    assert cap.supports("supports_lineage") is True
    assert cap.supports("remote") is False
    assert cap.supports("unknown_feature") is False


def test_tracking_protocol_version_metadata_only(tmp_path: Path) -> None:
    from aiodoo_training.domain.tracking_policies import TRACKING_PROTOCOL_VERSION

    root = tmp_path / "tracking"
    policy = to_tracking_policy(
        parse_tracking_config({"backend": "null", "enabled": True, "root_dir": str(root)})
    )
    builder = (
        TrackingBuilder()
        .with_policy(policy)
        .with_root(root)
        .with_identity(experiment_id=ExperimentId(value="e3"), name="n")
    )
    ctx = builder.build_context()
    assert ctx.experiment_session.tracking_protocol_version == TRACKING_PROTOCOL_VERSION
    assert ctx.run_record.tracking_protocol_version == TRACKING_PROTOCOL_VERSION
    before = build_provenance(config_fingerprint="cfg", model_fingerprint="m")
    coord = TrackingCoordinator(
        tracker=TrackerFactory().create("null"),
        context=ctx,
        store=TrackingStore(root),
    )
    coord.open()
    coord.complete()
    run_json = root / "experiments" / "e3" / "runs" / ctx.run_record.run_id.value / "run.json"
    assert run_json.is_file()
    import json

    payload = json.loads(run_json.read_text(encoding="utf-8"))
    assert payload["tracking_protocol_version"] == TRACKING_PROTOCOL_VERSION
    after = build_provenance(config_fingerprint="cfg", model_fingerprint="m")
    assert before.digest == after.digest


def test_cli_profiles() -> None:
    profile = to_cli_profile(parse_cli_config({"profile": "ci"}))
    assert profile.name is CLIProfileName.CI
    assert profile.output == "json"
    ctx = CommandBuilder().with_profile("verbose").with_json().build()
    assert ctx.json_output is True


def test_command_registry_doctor() -> None:
    reg = build_default_registry()
    ctx = CommandBuilder().build()
    assert reg.dispatch("doctor", ctx) == 0


def test_cmd_doctor_exit() -> None:
    assert cmd_doctor() == 0


def test_tracking_on_off_does_not_change_progress_objects() -> None:
    """Observational isolation smoke: MetricSnapshot equality independent of tracker."""
    snap = MetricSnapshot(name="loss", value=0.1, step=3)
    t1 = TrackerFactory().create("null")
    t2 = TrackerFactory().create("null")
    t1.log_metrics([snap])
    t2.log_metrics([snap])
    t1.close()
    t2.close()
    assert snap == MetricSnapshot(name="loss", value=0.1, step=3)


def test_catalog_and_run_index(tmp_path: Path) -> None:
    root = tmp_path / "tracking"
    policy = to_tracking_policy(
        parse_tracking_config({"backend": "null", "enabled": True, "root_dir": str(root)})
    )
    builder = (
        TrackingBuilder()
        .with_policy(policy)
        .with_root(root)
        .with_identity(experiment_id=ExperimentId(value="e2"), name="n")
    )
    ctx = builder.build_context()
    coord = TrackingCoordinator(
        tracker=TrackerFactory().create("null"),
        context=ctx,
        store=TrackingStore(root),
    )
    coord.open()
    coord.complete()
    from aiodoo_training.tracking import ExperimentCatalog, MetadataStore, RunIndex

    meta = MetadataStore(root)
    assert ExperimentCatalog(meta).get(ExperimentId(value="e2")) is not None
    assert RunIndex(meta).get(ctx.run_record.run_id) is not None
