"""Strongly typed identity value objects."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentId:
    """
    Deterministic identity of an experiment definition.

    Derived from canonical config hash, dataset digests, and version fingerprints.
    Shared across resume attempts of the same experiment.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("ExperimentId.value must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class RunId:
    """
    Unique identity of a single process execution.

    Distinct from ExperimentId: resumes share ExperimentId but mint a new RunId.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("RunId.value must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class StageName:
    """Stable name of a pipeline stage instance."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("StageName.value must be a non-empty string.")
