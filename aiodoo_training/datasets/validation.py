"""Dataset validation against protocol expectations.

Structural checks (required top-level keys, manifest protocol version) stay
training-owned: they operate on the raw record shape aiodoo-datasets writes
to disk, which is not itself an `aiodoo_contract` schema. Once a record's
dataset type has a canonical contract projection, this validator also
projects a sample of records and runs them through
`aiodoo_contract.validators.ContractValidator` — the same schema/capability
validation the contract mandates for every consumer — so a malformed
dataset fails fast during validation rather than surfacing later as a
`ContractAdapterError` mid-training.

Production safety: BenchmarkCatalog artifacts and any record with
``metadata.training_forbidden=true`` are rejected for LoRA training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aiodoo_contract.validators import ContractValidator

from aiodoo_training.contract.adapters import (
    SUPPORTED_CAPABILITIES,
    ContractAdapterError,
    project_record,
)
from aiodoo_training.datasets.reader import ProtocolRecordReader
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.exceptions import DomainError

REQUIRED_FIELDS: dict[DatasetType, frozenset[str]] = {
    DatasetType.PLANNER: frozenset({"instruction", "output", "metadata"}),
    DatasetType.CODING: frozenset({"instruction", "output", "metadata"}),
    DatasetType.REPAIR: frozenset({"instruction", "output", "metadata"}),
    DatasetType.EXECUTION: frozenset({"instruction", "output", "metadata"}),
    DatasetType.CONVERSATION: frozenset({"instruction", "output", "metadata"}),
    DatasetType.CONTEXT: frozenset({"id", "query", "metadata"}),
    DatasetType.APPROVAL: frozenset({"review_id", "decision", "metadata"}),
    # Evaluation v2 judgment SFT (EvaluationRequest → EvaluationResponse grain).
    DatasetType.EVALUATION: frozenset(
        {
            "record_id",
            "candidate_id",
            "evaluation_case_key",
            "capability_under_test",
            "candidate",
            "verdict",
            "metadata",
        }
    ),
    DatasetType.MIXED: frozenset(),
}

_BENCHMARK_CATALOG_ERROR = (
    "BenchmarkCatalog is certification-only and cannot be used for LoRA training"
)


class DatasetValidator:
    """Fail-fast validation for DatasetRef targets before training consume."""

    def __init__(
        self,
        reader: ProtocolRecordReader | None = None,
        *,
        contract_validator: ContractValidator | None = None,
    ) -> None:
        self._reader = reader or ProtocolRecordReader()
        self._contract_validator = contract_validator or ContractValidator()

    def validate_ref(self, ref: DatasetRef, *, sample_limit: int = 32) -> None:
        """
        Validate that ``ref.path`` exists, is readable JSONL, and matches type rules.

        Raises:
            DomainError: on any contract violation.
        """
        path = Path(ref.path)
        if not path.exists():
            raise DomainError(f"Dataset path does not exist: {path}")
        if ref.protocol_version.strip() == "":
            raise DomainError("DatasetRef.protocol_version must be non-empty.")

        self._reject_benchmark_catalog_path(path)

        manifest_path = path.with_name(path.stem + "_manifest.json")
        if not manifest_path.exists():
            # Alternate convention: sibling manifest.json next to dataset
            alt = path.with_name(path.stem.replace("_dataset", "") + "_manifest.json")
            if alt.exists():
                manifest_path = alt

        if manifest_path.exists():
            self._validate_manifest(manifest_path, ref)

        required = REQUIRED_FIELDS.get(ref.dataset_type, frozenset())
        capability = ref.dataset_type.value
        has_contract_projection = capability in SUPPORTED_CAPABILITIES
        seen = 0
        for record in self._reader.iter_records(path):
            seen += 1
            self._reject_forbidden_or_catalog_record(path, record)
            missing = required - record.keys()
            if missing:
                raise DomainError(
                    f"Dataset {path} record missing required fields "
                    f"for {ref.dataset_type.value}: {sorted(missing)}"
                )
            if has_contract_projection:
                self._validate_contract_projection(path, capability, record)
            if seen >= sample_limit:
                break
        if seen == 0:
            raise DomainError(f"Dataset is empty: {path}")

    @staticmethod
    def _reject_benchmark_catalog_path(path: Path) -> None:
        if "benchmark_catalog" in path.name.lower():
            raise DomainError(f"Dataset {path}: {_BENCHMARK_CATALOG_ERROR}.")

    @staticmethod
    def _reject_forbidden_or_catalog_record(path: Path, record: dict[str, Any]) -> None:
        metadata = record.get("metadata")
        if isinstance(metadata, dict) and metadata.get("training_forbidden") is True:
            raise DomainError(
                f"Dataset {path} record is marked metadata.training_forbidden=true "
                "and cannot be used for LoRA training."
            )

        catalog = record.get("catalog")
        if isinstance(catalog, dict) and (
            "suites" in catalog or "catalog_id" in catalog or "catalog_name" in catalog
        ):
            raise DomainError(
                f"Dataset {path} contains a BenchmarkCatalog record: "
                f"{_BENCHMARK_CATALOG_ERROR}."
            )

    def _validate_contract_projection(
        self, path: Path, capability: str, record: dict[str, Any]
    ) -> None:
        """Project ``record`` onto its contract shape and run `ContractValidator` on it.

        Raises:
            DomainError: if the record cannot be projected, or the
                projected request/response fails contract validation.
        """
        try:
            projection = project_record(capability, record)
        except ContractAdapterError as exc:
            raise DomainError(
                f"Dataset {path} record cannot be projected onto the aiodoo_contract "
                f"{capability!r} capability: {exc}"
            ) from exc

        result = self._contract_validator.validate_request(projection.request)
        result = result.merge(self._contract_validator.validate_response(projection.response))
        if not result:
            issues = "; ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
            raise DomainError(
                f"Dataset {path} record failed aiodoo_contract validation "
                f"for capability {capability!r}: {issues}"
            )

    def _validate_manifest(self, manifest_path: Path, ref: DatasetRef) -> None:
        try:
            data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DomainError(f"Invalid dataset manifest {manifest_path}: {exc}") from exc

        dataset_name = str(data.get("dataset_name", "")).strip().lower()
        if dataset_name == "benchmark_catalog" or "benchmark_catalog" in dataset_name:
            raise DomainError(
                f"Manifest {manifest_path} describes a BenchmarkCatalog artifact: "
                f"{_BENCHMARK_CATALOG_ERROR}."
            )

        protocol = str(data.get("protocol_version", "")).strip()
        if protocol and protocol != ref.protocol_version:
            raise DomainError(
                f"Protocol version mismatch for {ref.path}: "
                f"ref={ref.protocol_version!r} manifest={protocol!r}"
            )
