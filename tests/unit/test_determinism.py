"""Determinism foundation tests."""

import pytest

from aiodoo_training import __version__
from aiodoo_training.determinism import FingerprintService, SeedManager
from aiodoo_training.exceptions import DeterminismError


def test_seed_manager_rejects_invalid_seeds() -> None:
    with pytest.raises(DeterminismError):
        SeedManager(seed=-1)
    with pytest.raises(DeterminismError):
        SeedManager(seed=True)  # type: ignore[arg-type]


def test_seed_python_and_snapshot_roundtrip() -> None:
    manager = SeedManager(seed=123)
    manager.seed_python()
    snap = manager.snapshot()
    manager.set_seed(999)
    manager.restore(snap)
    assert manager.seed == 123


def test_deferred_rng_backends_raise() -> None:
    manager = SeedManager(seed=1)
    with pytest.raises(NotImplementedError, match="NumPy"):
        manager.seed_numpy()
    with pytest.raises(NotImplementedError, match="Torch"):
        manager.seed_torch()
    with pytest.raises(NotImplementedError, match="CUDA"):
        manager.seed_cuda()


def test_experiment_fingerprint_is_stable() -> None:
    service = FingerprintService()
    data = {"name": "demo", "seed": 7}
    first = service.experiment_fingerprint(data, package_version=__version__)
    second = service.experiment_fingerprint(data, package_version=__version__)
    assert first.digest == second.digest
    assert first.experiment_id == second.experiment_id
    assert first.packages.digest == second.packages.digest
    assert first.environment.digest


def test_environment_excluded_from_default_identity() -> None:
    service = FingerprintService()
    data = {"name": "demo"}
    without_env = service.experiment_fingerprint(data, package_version="0.1.0")
    with_env = service.experiment_fingerprint(
        data,
        package_version="0.1.0",
        include_environment=True,
    )
    assert without_env.digest != with_env.digest
