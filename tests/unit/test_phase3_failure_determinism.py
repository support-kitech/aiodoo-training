"""Phase 3 failure-path and boundary extras."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiodoo_training.bootstrap import bootstrap_phase3
from aiodoo_training.domain.enums import TrainingStatus
from aiodoo_training.exceptions import DomainError
from aiodoo_training.infrastructure.stub.trainer import StubTrainerBackend
from aiodoo_training.training.engine import build_stub_training_context, make_stub_experiment_config


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase3(overwrite=True)


def test_two_uninterrupted_runs_match(tmp_path: Path) -> None:
    """Determinism: same config/seed → same loss sequence."""
    from aiodoo_training.training.engine import run_stub_train

    losses = []
    for name in ("a", "b"):
        cfg = make_stub_experiment_config(
            name="det",
            seed=123,
            max_steps=5,
            save_steps=100,
            output_dir=tmp_path / name,
        )
        ctx = build_stub_training_context(config=cfg)
        _, progress = run_stub_train(ctx)
        losses.append([(m.step, m.value) for m in progress.metrics if m.name == "loss"])
    assert losses[0] == losses[1]


def test_stub_trainer_requires_dict_framework_model(tmp_path: Path) -> None:
    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=1, save_steps=100)
    ctx = build_stub_training_context(config=cfg)
    trainer = StubTrainerBackend().bind(ctx)
    # Corrupt carrier
    from aiodoo_training.infrastructure.model_handles import require_trainable_carrier

    carrier = require_trainable_carrier(ctx.model)
    carrier.framework_model = "not-a-dict"  # type: ignore[assignment]
    with pytest.raises(DomainError, match="dict"):
        trainer.train(cfg, ctx.model, ctx.execution)


def test_paused_status_on_stop_after(tmp_path: Path) -> None:
    from aiodoo_training.training.engine import run_stub_train

    cfg = make_stub_experiment_config(output_dir=tmp_path, max_steps=10, save_steps=100)
    ctx = build_stub_training_context(config=cfg, stop_after_steps=2)
    _, progress = run_stub_train(ctx)
    assert progress.status is TrainingStatus.PAUSED
    assert progress.global_step == 2
