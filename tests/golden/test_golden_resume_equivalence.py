"""Golden: resumed stub training matches uninterrupted progression (§12.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase3
from aiodoo_training.domain.enums import TrainingStatus
from aiodoo_training.infrastructure.model_handles import require_trainable_carrier
from aiodoo_training.training.engine import (
    build_stub_training_context,
    make_stub_experiment_config,
    resume_from_checkpoint,
    run_stub_train,
)


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase3(overwrite=True)


def _loss_sequence(progress) -> list[tuple[int, float]]:
    return [(m.step, m.value) for m in progress.metrics if m.name == "loss"]


def test_golden_resume_equivalence(tmp_path: Path) -> None:
    """
    Uninterrupted N steps vs checkpoint@K then resume to N must match:

    - global_step / epoch
    - loss sequence for steps K+1…N
    - final TrainingProgress metrics
    - DatasetSession cursor
    - TrainingSession step
    - weight vector / RNG seed continuity
    """
    seed = 42
    n_steps = 10
    k_steps = 4
    save_steps = 4

    # --- Uninterrupted reference ---
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    ref_cfg = make_stub_experiment_config(
        name="golden-ref",
        seed=seed,
        max_steps=n_steps,
        save_steps=save_steps,
        output_dir=ref_dir,
    )
    ref_ctx = build_stub_training_context(config=ref_cfg)
    ref_ctx, ref_progress = run_stub_train(ref_ctx)
    assert ref_progress.status is TrainingStatus.COMPLETED
    assert ref_progress.global_step == n_steps
    ref_losses = _loss_sequence(ref_progress)
    ref_weights = list(require_trainable_carrier(ref_ctx.model).framework_model["weights"])
    ref_session = ref_ctx.training_session
    ref_dataset = ref_ctx.dataset_session
    ref_history = (
        list(ref_ctx.metric_collector.history.snapshots)
        if ref_ctx.metric_collector is not None
        else []
    )

    # --- Interrupted at K (after checkpoint-K published) ---
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_cfg = make_stub_experiment_config(
        name="golden-ref",  # same experiment id/name for resume validation
        seed=seed,
        max_steps=n_steps,
        save_steps=save_steps,
        output_dir=run_dir,
    )
    run_ctx = build_stub_training_context(config=run_cfg, stop_at_step=k_steps)
    run_ctx, paused = run_stub_train(run_ctx)
    assert paused.status is TrainingStatus.PAUSED
    assert paused.global_step == k_steps
    ckpt = run_dir / f"checkpoint-{k_steps}"
    assert ckpt.is_dir()
    assert (ckpt / "manifest.json").is_file()

    # Simulate process termination: new context + resume coordinator path
    cont_cfg = make_stub_experiment_config(
        name="golden-ref",
        seed=seed,
        max_steps=n_steps,
        save_steps=save_steps,
        output_dir=run_dir,
        resume_from=ckpt,
    )
    cont_ctx, bundle, cont_progress = resume_from_checkpoint(
        config=cont_cfg, checkpoint_path=ckpt
    )

    assert bundle.manifest.global_step == k_steps
    assert cont_progress.status is TrainingStatus.COMPLETED
    assert cont_progress.global_step == n_steps
    assert cont_progress.epoch == pytest.approx(ref_progress.epoch)

    # Loss sequence: first segment + resumed segment == uninterrupted
    paused_losses = _loss_sequence(paused)
    resumed_losses = _loss_sequence(cont_progress)
    combined = paused_losses + resumed_losses
    assert combined == ref_losses

    # Weights / session / dataset cursor
    cont_weights = list(require_trainable_carrier(cont_ctx.model).framework_model["weights"])
    assert cont_weights == pytest.approx(ref_weights)
    assert cont_ctx.training_session.global_step == ref_session.global_step
    assert cont_ctx.dataset_session.examples_seen == ref_dataset.examples_seen
    assert cont_ctx.dataset_session.example_index == ref_dataset.example_index

    # Checkpoint / resume fingerprints present
    assert bundle.manifest.checkpoint_fingerprint
    assert bundle.training_session.checkpoint_fingerprint == bundle.manifest.checkpoint_fingerprint

    # TrainingHistory from resumed collector includes post-resume losses
    if cont_ctx.metric_collector is not None:
        hist_steps = [s.step for s in cont_ctx.metric_collector.history.snapshots]
        assert hist_steps == [m.step for m in cont_progress.metrics if m.name == "loss"]

    # RNG seed restored from checkpoint
    assert cont_ctx.rng.snapshot()["seed"] == seed

    # Reference history length equals full uninterrupted metric trail when collector used
    assert len(ref_history) == n_steps
