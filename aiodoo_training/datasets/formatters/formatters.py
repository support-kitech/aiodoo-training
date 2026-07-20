"""Protocol → TrainingExample formatters for all dataset types.

Six of the eight dataset types (planner, coding, repair, execution,
conversation, approval) map onto a capability the Capability Contract
defines a shape for. Those formatters build their `TrainingExample`
exclusively through :mod:`aiodoo_training.contract` — projecting the raw
record onto `aiodoo_contract.schemas`, rendering the prompt with
`aiodoo_contract.prompts.CapabilityPromptBuilder`, and teaching the model
the canonical `CapabilityResponse` JSON as its label. Training does not
hand-assemble instruction/context strings for these six (ADR-0003).

``context`` and ``evaluation`` have no contract projection — the same gap
aiodoo-datasets documents for its own contract adapter (evaluation's
BenchmarkCatalog domain does not map onto EvaluationRequest/Response;
``context`` is not itself a capability). Those two keep the prior
dataset-specific formatting; see ``CONTRACT_ADOPTION.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aiodoo_training.contract.adapters import ContractAdapterError
from aiodoo_training.contract.prompt_bridge import build_training_example
from aiodoo_training.datasets.formatters.base import BaseFormatter, _json_dump, user_assistant
from aiodoo_training.datasets.mixing import stable_example_id
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.examples import TrainingExample
from aiodoo_training.exceptions import DomainError


class _ContractFormatter(BaseFormatter):
    """Shared `_format` for the six dataset types with a contract projection.

    Subclasses only set :attr:`dataset_type`; the capability name passed to
    :mod:`aiodoo_training.contract` is always ``dataset_type.value`` because
    every dataset type this class serves is named identically to its
    capability (see ``aiodoo_contract.schemas.enums.CapabilityName``).
    """

    def _format(self, record: Mapping[str, Any]) -> TrainingExample:
        capability = self.dataset_type.value
        example_id = stable_example_id(capability, record, 0)
        try:
            return build_training_example(
                dataset_type=self.dataset_type,
                capability=capability,
                record=dict(record),
                example_id=example_id,
            )
        except ContractAdapterError as exc:
            raise DomainError(
                f"{type(self).__name__} could not project a record onto the "
                f"aiodoo_contract {capability!r} capability: {exc}"
            ) from exc


class PlannerFormatter(_ContractFormatter):
    dataset_type = DatasetType.PLANNER


class CodingFormatter(_ContractFormatter):
    dataset_type = DatasetType.CODING


class RepairFormatter(_ContractFormatter):
    dataset_type = DatasetType.REPAIR


class ExecutionFormatter(_ContractFormatter):
    dataset_type = DatasetType.EXECUTION


class ConversationFormatter(_ContractFormatter):
    dataset_type = DatasetType.CONVERSATION


class ApprovalFormatter(_ContractFormatter):
    dataset_type = DatasetType.APPROVAL


class ContextFormatter(BaseFormatter):
    """No contract projection: `context` is not itself a capability (see module docstring)."""

    dataset_type = DatasetType.CONTEXT

    def _format(self, record: Mapping[str, Any]) -> TrainingExample:
        query = str(record.get("query", ""))
        assistant_payload = {
            "artifacts": record.get("artifacts"),
            "graph": record.get("graph"),
        }
        return user_assistant(
            self.dataset_type,
            record,
            user_text=query,
            assistant_text=_json_dump(assistant_payload),
        )


class EvaluationFormatter(BaseFormatter):
    """No contract projection: evaluation's BenchmarkCatalog domain does not map onto

    `EvaluationRequest`/`EvaluationResponse` (see module docstring; matches the
    same documented gap in aiodoo-datasets' contract adapter).
    """

    dataset_type = DatasetType.EVALUATION

    def _format(self, record: Mapping[str, Any]) -> TrainingExample:
        catalog = record.get("catalog")
        user = f"Evaluate using the following catalog:\n{_json_dump(catalog)}"
        return user_assistant(
            self.dataset_type,
            record,
            user_text=user,
            assistant_text=_json_dump({"evaluation_id": record.get("evaluation_id")}),
        )
