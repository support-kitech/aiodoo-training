"""Determinism and fingerprint foundations."""

from __future__ import annotations

import hashlib
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiodoo_training.config.system import ConfigHasher
from aiodoo_training.domain.identifiers import ExperimentId
from aiodoo_training.exceptions import DeterminismError


@dataclass(frozen=True, slots=True)
class ConfigFingerprint:
    """Fingerprint of a composed (portable) experiment configuration."""

    digest: str
    canonical_json: str


@dataclass(frozen=True, slots=True)
class DatasetFingerprint:
    """Fingerprint of dataset artifacts (content digests arrive in Phase 1)."""

    digest: str
    references: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PackageFingerprint:
    """
    Fingerprint of installed training-relevant packages.

    Phase 0 records only the aiodoo-training version string. Later phases may
    add torch / transformers / peft versions via ``extra``.
    """

    digest: str
    package_version: str
    extra: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class VersionFingerprint:
    """Fingerprint of Python and repository versions."""

    digest: str
    python_version: str
    package_version: str
    extra: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class EnvironmentFingerprint:
    """
    Fingerprint of the execution environment.

    CPU/GPU device names and CUDA driver details are deferred until ML backends
    exist. Structure is stable for forward-compatible extension.
    """

    digest: str
    platform: str
    machine: str
    processor: str
    extra: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ExperimentFingerprint:
    """
    Aggregate deterministic identity inputs for an experiment.

    ``ExperimentId`` is derived from the composite identity digest.
    """

    config: ConfigFingerprint
    datasets: DatasetFingerprint
    versions: VersionFingerprint
    packages: PackageFingerprint
    environment: EnvironmentFingerprint
    digest: str
    experiment_id: ExperimentId


class SeedManager:
    """
    Owns the global experiment seed and future multi-backend RNG seeding.

    Phase 0 implements Python's ``random`` module only. Extension points for
    NumPy, Torch, and CUDA are declared without importing those libraries.
    """

    def __init__(self, seed: int = 42) -> None:
        self.set_seed(seed)

    @property
    def seed(self) -> int:
        """Current non-negative experiment seed."""
        return self._seed

    def set_seed(self, seed: int) -> None:
        """Validate and store ``seed``."""
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise DeterminismError("Seed must be a non-negative integer.")
        self._seed = seed

    def seed_all(self) -> None:
        """
        Seed all available RNG backends.

        Phase 0: Python ``random`` only.
        Later phases: also NumPy, Torch CPU, and CUDA where present.
        """
        self.seed_python()
        # Deferred backends — keep call order stable for future phases:
        # self.seed_numpy()
        # self.seed_torch()
        # self.seed_cuda()

    def seed_python(self) -> None:
        """Seed the Python standard-library ``random`` module."""
        random.seed(self._seed)

    def seed_numpy(self) -> None:
        """Seed NumPy RNGs (deferred — no NumPy dependency in Phase 0)."""
        raise NotImplementedError("seed_numpy requires NumPy (deferred to a later phase).")

    def seed_torch(self) -> None:
        """Seed Torch CPU RNGs (deferred — no Torch dependency in Phase 0)."""
        raise NotImplementedError("seed_torch requires Torch (deferred to a later phase).")

    def seed_cuda(self) -> None:
        """Seed CUDA RNGs (deferred — no CUDA dependency in Phase 0)."""
        raise NotImplementedError("seed_cuda requires Torch CUDA (deferred to a later phase).")

    def snapshot(self) -> dict[str, object]:
        """
        Capture RNG state for checkpoint restore.

        Phase 0 returns Python ``random`` state only.
        """
        return {
            "seed": self._seed,
            "python_random": random.getstate(),
        }

    def restore(self, state: dict[str, object]) -> None:
        """Restore RNG state previously produced by :meth:`snapshot`."""
        seed = state.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise DeterminismError("Invalid seed in RNG snapshot.")
        self._seed = seed
        python_state = state.get("python_random")
        if python_state is not None:
            random.setstate(python_state)  # type: ignore[arg-type]


class FingerprintService:
    """Constructs deterministic fingerprints from config and environment metadata."""

    def __init__(self, hasher: ConfigHasher | None = None) -> None:
        self._hasher = hasher or ConfigHasher()

    def config_fingerprint(self, data: dict[str, Any]) -> ConfigFingerprint:
        """Fingerprint a portable composed configuration mapping."""
        canonical = self._hasher.canonical_json(data)
        return ConfigFingerprint(digest=self._hasher.hash(data), canonical_json=canonical)

    def dataset_fingerprint(self, paths: tuple[Path, ...] = ()) -> DatasetFingerprint:
        """
        Phase 0 placeholder: hashes resolved path strings only.

        Phase 1 will hash dataset file contents and manifests.
        """
        normalized = tuple(sorted(str(p.resolve()) for p in paths))
        payload = "\n".join(normalized).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return DatasetFingerprint(digest=digest, references=normalized)

    def package_fingerprint(
        self,
        package_version: str,
        extra: tuple[tuple[str, str], ...] = (),
    ) -> PackageFingerprint:
        """Fingerprint repository / future ML package versions."""
        parts = [f"aiodoo-training={package_version}"]
        parts.extend(f"{name}={version}" for name, version in sorted(extra))
        digest = hashlib.sha256(";".join(parts).encode("utf-8")).hexdigest()
        return PackageFingerprint(
            digest=digest,
            package_version=package_version,
            extra=tuple(sorted(extra)),
        )

    def version_fingerprint(self, package_version: str) -> VersionFingerprint:
        """Fingerprint Python runtime and repository version."""
        python_version = sys.version.split()[0]
        material = f"python={python_version};pkg={package_version}".encode()
        digest = hashlib.sha256(material).hexdigest()
        return VersionFingerprint(
            digest=digest,
            python_version=python_version,
            package_version=package_version,
        )

    def environment_fingerprint(self) -> EnvironmentFingerprint:
        """Fingerprint OS/hardware identity (GPU details deferred)."""
        platform_name = platform.platform()
        machine = platform.machine()
        processor = platform.processor() or "unknown"
        material = f"{platform_name}|{machine}|{processor}".encode()
        digest = hashlib.sha256(material).hexdigest()
        return EnvironmentFingerprint(
            digest=digest,
            platform=platform_name,
            machine=machine,
            processor=processor,
        )

    def experiment_fingerprint(
        self,
        config_data: dict[str, Any],
        *,
        package_version: str,
        dataset_paths: tuple[Path, ...] = (),
        include_environment: bool = False,
        package_extra: tuple[tuple[str, str], ...] = (),
    ) -> ExperimentFingerprint:
        """
        Build an experiment fingerprint.

        Environment is excluded from the identity digest by default so the same
        experiment config yields the same ExperimentId across machines.
        """
        config_fp = self.config_fingerprint(config_data)
        dataset_fp = self.dataset_fingerprint(dataset_paths)
        version_fp = self.version_fingerprint(package_version)
        package_fp = self.package_fingerprint(package_version, package_extra)
        env_fp = self.environment_fingerprint()

        identity_material = (
            f"{config_fp.digest}:{dataset_fp.digest}:{version_fp.digest}:{package_fp.digest}"
        )
        if include_environment:
            identity_material = f"{identity_material}:{env_fp.digest}"
        digest = hashlib.sha256(identity_material.encode("utf-8")).hexdigest()
        experiment_id = ExperimentId(value=f"exp_{digest[:16]}")
        return ExperimentFingerprint(
            config=config_fp,
            datasets=dataset_fp,
            versions=version_fp,
            packages=package_fp,
            environment=env_fp,
            digest=digest,
            experiment_id=experiment_id,
        )
