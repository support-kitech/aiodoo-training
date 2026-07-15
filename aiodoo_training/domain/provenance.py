"""Phase 6 provenance snapshots — portable, fingerprint-safe (no volatile fields)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def _canonical(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshot:
    """Portable composed config digest + selected fragment keys."""

    config_fingerprint: str
    schema_version: str
    fragment_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """ExecutionEnvironment digest summary (no secrets)."""

    execution_digest: str
    device_kind: str = "cpu"
    precision: str = "fp32"


@dataclass(frozen=True, slots=True)
class DependencySnapshot:
    """Optional package versions (absent packages omitted)."""

    packages: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class HardwareSnapshot:
    """HardwareCapabilities summary from ResourcePlanner resolution."""

    accelerator: str = "none"
    memory_hint: str | None = None


@dataclass(frozen=True, slots=True)
class SoftwareSnapshot:
    """aiodoo-training version + protocol versions."""

    package_version: str
    training_protocol_version: str = "1"
    artifact_protocol_version: str = "1"


@dataclass(frozen=True, slots=True)
class ExperimentProvenance:
    """Aggregation of portable snapshots + identity fingerprints."""

    configuration: ConfigurationSnapshot
    environment: EnvironmentSnapshot
    dependencies: DependencySnapshot
    hardware: HardwareSnapshot
    software: SoftwareSnapshot
    dataset_fingerprint: str = ""
    model_fingerprint: str = ""
    adapter_fingerprint: str = ""

    @property
    def digest(self) -> str:
        """Stable digest excluding wall-clock fields."""
        return _sha(
            self.configuration.config_fingerprint,
            self.configuration.schema_version,
            ",".join(self.configuration.fragment_keys),
            self.environment.execution_digest,
            self.environment.device_kind,
            self.environment.precision,
            _canonical(list(self.dependencies.packages)),
            self.hardware.accelerator,
            self.hardware.memory_hint or "",
            self.software.package_version,
            self.software.training_protocol_version,
            self.software.artifact_protocol_version,
            self.dataset_fingerprint,
            self.model_fingerprint,
            self.adapter_fingerprint,
        )
