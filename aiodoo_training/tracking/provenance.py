"""Provenance builders — portable digests without volatile fields."""

from __future__ import annotations

from aiodoo_training import __version__
from aiodoo_training.domain.provenance import (
    ConfigurationSnapshot,
    DependencySnapshot,
    EnvironmentSnapshot,
    ExperimentProvenance,
    HardwareSnapshot,
    SoftwareSnapshot,
)


def build_provenance(
    *,
    config_fingerprint: str,
    schema_version: str = "1",
    fragment_keys: tuple[str, ...] = (),
    execution_digest: str = "",
    device_kind: str = "cpu",
    precision: str = "fp32",
    packages: tuple[tuple[str, str], ...] = (),
    accelerator: str = "none",
    memory_hint: str | None = None,
    package_version: str | None = None,
    dataset_fingerprint: str = "",
    model_fingerprint: str = "",
    adapter_fingerprint: str = "",
) -> ExperimentProvenance:
    return ExperimentProvenance(
        configuration=ConfigurationSnapshot(
            config_fingerprint=config_fingerprint,
            schema_version=schema_version,
            fragment_keys=fragment_keys,
        ),
        environment=EnvironmentSnapshot(
            execution_digest=execution_digest,
            device_kind=device_kind,
            precision=precision,
        ),
        dependencies=DependencySnapshot(packages=packages),
        hardware=HardwareSnapshot(accelerator=accelerator, memory_hint=memory_hint),
        software=SoftwareSnapshot(package_version=package_version or __version__),
        dataset_fingerprint=dataset_fingerprint,
        model_fingerprint=model_fingerprint,
        adapter_fingerprint=adapter_fingerprint,
    )
