"""Checkpoint / resume / corruption / ResumePolicy tests (CPU stub)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase3
from aiodoo_training.domain.training_policies import ResumePolicy
from aiodoo_training.exceptions import CheckpointCorruption, IncompatibleResume
from aiodoo_training.training.checkpoint_manager import ResumeValidationContext
from aiodoo_training.training.engine import (
    build_stub_training_context,
    make_stub_experiment_config,
    resume_from_checkpoint,
    run_stub_train,
)
from aiodoo_training.training.resume import ResumeCoordinator


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase3(overwrite=True)


def test_atomic_checkpoint_save_and_list(tmp_path: Path) -> None:
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=4, save_steps=2)
    ctx = build_stub_training_context(config=cfg)
    ctx, progress = run_stub_train(ctx)
    assert progress.global_step == 4
    assert ctx.checkpoint_manager is not None
    handles = ctx.checkpoint_manager.list_checkpoints(tmp_path)
    assert len(handles) >= 1
    ckpt = tmp_path / "checkpoint-2"
    assert ckpt.is_dir()
    assert (ckpt / "manifest.json").is_file()
    assert (ckpt / "rng.json").is_file()
    assert (ckpt / "dataset_session.json").is_file()
    assert (ckpt / "weights.json").is_file()
    assert not any(p.name.startswith(".tmp-") for p in tmp_path.iterdir())


def test_resume_policy_strict_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=2, save_steps=2)
    ctx = build_stub_training_context(config=cfg)
    ctx, _ = run_stub_train(ctx)
    ckpt = tmp_path / "checkpoint-2"
    coordinator = ResumeCoordinator(
        checkpoint_store=ctx.checkpoint_store,
        rng=ctx.rng,
        checkpoint_manager=ctx.checkpoint_manager,
    )
    expected = ResumeValidationContext(
        experiment_id=ctx.training_session.experiment_id,
        model_fingerprint="wrong-model",
        adapter_fingerprint=ctx.adapter_fingerprint,
        config_fingerprint=ctx.config_fingerprint,
        execution_digest=ctx.execution_digest,
        trainer_backend_key="stub",
    )
    with pytest.raises(IncompatibleResume, match="model_fingerprint"):
        coordinator.load_and_validate(
            ckpt,
            expected=expected,
            policy=ResumePolicy.STRICT,
            training_session=ctx.training_session,
        )


def test_resume_policy_warn_allows_model_fingerprint_mismatch(tmp_path: Path) -> None:
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=2, save_steps=2)
    ctx = build_stub_training_context(config=cfg)
    ctx, _ = run_stub_train(ctx)
    ckpt = tmp_path / "checkpoint-2"
    coordinator = ResumeCoordinator(
        checkpoint_store=ctx.checkpoint_store,
        rng=ctx.rng,
        checkpoint_manager=ctx.checkpoint_manager,
    )
    expected = ResumeValidationContext(
        experiment_id=ctx.training_session.experiment_id,
        model_fingerprint="wrong-model",
        trainer_backend_key="stub",
    )
    bundle = coordinator.load_and_validate(
        ckpt,
        expected=expected,
        policy=ResumePolicy.WARN,
        training_session=ctx.training_session,
    )
    assert any("model_fingerprint" in w for w in bundle.warnings)


def test_corruption_missing_manifest(tmp_path: Path) -> None:
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=2, save_steps=2)
    ctx = build_stub_training_context(config=cfg)
    ctx, _ = run_stub_train(ctx)
    ckpt = tmp_path / "checkpoint-2"
    (ckpt / "manifest.json").unlink()
    coordinator = ResumeCoordinator(
        checkpoint_store=ctx.checkpoint_store,
        rng=ctx.rng,
        checkpoint_manager=ctx.checkpoint_manager,
    )
    expected = ResumeValidationContext(
        experiment_id=ctx.training_session.experiment_id,
        trainer_backend_key="stub",
    )
    with pytest.raises(CheckpointCorruption, match="manifest"):
        coordinator.load_and_validate(
            ckpt,
            expected=expected,
            policy=ResumePolicy.STRICT,
            training_session=ctx.training_session,
        )


def test_corruption_missing_weights(tmp_path: Path) -> None:
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=2, save_steps=2)
    ctx = build_stub_training_context(config=cfg)
    ctx, _ = run_stub_train(ctx)
    ckpt = tmp_path / "checkpoint-2"
    (ckpt / "weights.json").unlink()
    coordinator = ResumeCoordinator(
        checkpoint_store=ctx.checkpoint_store,
        rng=ctx.rng,
        checkpoint_manager=ctx.checkpoint_manager,
    )
    expected = ResumeValidationContext(
        experiment_id=ctx.training_session.experiment_id,
        trainer_backend_key="stub",
    )
    with pytest.raises(CheckpointCorruption):
        coordinator.load_and_validate(
            ckpt,
            expected=expected,
            policy=ResumePolicy.STRICT,
            training_session=ctx.training_session,
        )


def test_tmp_cleanup(tmp_path: Path) -> None:
    stale = tmp_path / ".tmp-deadbeef"
    stale.mkdir()
    (stale / "junk").write_text("x", encoding="utf-8")
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=2, save_steps=2)
    ctx = build_stub_training_context(config=cfg)
    assert ctx.checkpoint_manager is not None
    removed = ctx.checkpoint_manager.cleanup_incomplete_tmp_dirs(tmp_path)
    assert stale in removed
    assert not stale.exists()


def test_protocol_version_reject(tmp_path: Path) -> None:
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=2, save_steps=2)
    ctx = build_stub_training_context(config=cfg)
    ctx, _ = run_stub_train(ctx)
    ckpt = tmp_path / "checkpoint-2"
    manifest_path = ckpt / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["training_protocol_version"] = "999"
    # Fingerprint will also mismatch — recompute would need update; protocol checked first.
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    coordinator = ResumeCoordinator(
        checkpoint_store=ctx.checkpoint_store,
        rng=ctx.rng,
        checkpoint_manager=ctx.checkpoint_manager,
    )
    expected = ResumeValidationContext(
        experiment_id=ctx.training_session.experiment_id,
        trainer_backend_key="stub",
    )
    with pytest.raises(IncompatibleResume, match="training_protocol_version"):
        coordinator.load_and_validate(
            ckpt,
            expected=expected,
            policy=ResumePolicy.RELAXED,
            training_session=ctx.training_session,
        )


def test_resume_continues_steps(tmp_path: Path) -> None:
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=6, save_steps=3)
    ctx = build_stub_training_context(config=cfg, stop_at_step=3)
    ctx, paused = run_stub_train(ctx)
    assert paused.status.value == "paused"
    assert paused.global_step == 3
    cfg2 = make_stub_experiment_config(
        output_dir=tmp_path, max_steps=6, save_steps=3, resume_from=tmp_path / "checkpoint-3"
    )
    # Keep same fingerprints/name/seed
    cfg2 = make_stub_experiment_config(output_dir=tmp_path, max_steps=6, save_steps=3)
    _, bundle, progress = resume_from_checkpoint(
        config=cfg2, checkpoint_path=tmp_path / "checkpoint-3"
    )
    assert bundle.checkpoint.global_step == 3
    assert progress.global_step == 6
    assert progress.status.value == "completed"
