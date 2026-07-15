"""Protocol → TrainingExample formatters for all dataset types."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aiodoo_training.datasets.formatters.base import BaseFormatter, _json_dump, user_assistant
from aiodoo_training.domain.enums import DatasetType
from aiodoo_training.domain.examples import TrainingExample


class PlannerFormatter(BaseFormatter):
    dataset_type = DatasetType.PLANNER

    def _format(self, record: Mapping[str, Any]) -> TrainingExample:
        instruction = str(record.get("instruction", ""))
        input_text = record.get("input", "")
        user = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
        return user_assistant(
            self.dataset_type,
            record,
            user_text=user,
            assistant_text=_json_dump(record.get("output", {})),
        )


class CodingFormatter(BaseFormatter):
    dataset_type = DatasetType.CODING

    def _format(self, record: Mapping[str, Any]) -> TrainingExample:
        instruction = str(record.get("instruction", ""))
        context = record.get("context")
        user = instruction
        if context is not None:
            user = f"{instruction}\n\nContext:\n{_json_dump(context)}"
        return user_assistant(
            self.dataset_type,
            record,
            user_text=user,
            assistant_text=_json_dump(record.get("output", {})),
        )


class RepairFormatter(BaseFormatter):
    dataset_type = DatasetType.REPAIR

    def _format(self, record: Mapping[str, Any]) -> TrainingExample:
        instruction = str(record.get("instruction", ""))
        context = record.get("context")
        user = instruction
        if context is not None:
            user = f"{instruction}\n\nContext:\n{_json_dump(context)}"
        return user_assistant(
            self.dataset_type,
            record,
            user_text=user,
            assistant_text=_json_dump(record.get("output", {})),
        )


class ExecutionFormatter(BaseFormatter):
    dataset_type = DatasetType.EXECUTION

    def _format(self, record: Mapping[str, Any]) -> TrainingExample:
        instruction = str(record.get("instruction", ""))
        context = record.get("context")
        user = instruction
        if context is not None:
            user = f"{instruction}\n\nContext:\n{_json_dump(context)}"
        return user_assistant(
            self.dataset_type,
            record,
            user_text=user,
            assistant_text=_json_dump(record.get("output", {})),
        )


class ConversationFormatter(BaseFormatter):
    dataset_type = DatasetType.CONVERSATION

    def _format(self, record: Mapping[str, Any]) -> TrainingExample:
        instruction = str(record.get("instruction", ""))
        context = record.get("context")
        user = instruction
        if context is not None:
            user = f"{instruction}\n\nContext:\n{_json_dump(context)}"
        return user_assistant(
            self.dataset_type,
            record,
            user_text=user,
            assistant_text=_json_dump(record.get("output", {})),
        )


class ContextFormatter(BaseFormatter):
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


class ApprovalFormatter(BaseFormatter):
    dataset_type = DatasetType.APPROVAL

    def _format(self, record: Mapping[str, Any]) -> TrainingExample:
        evidence = record.get("evidence")
        findings = record.get("findings")
        user = (
            "Review the following findings and evidence, then decide.\n\n"
            f"Findings:\n{_json_dump(findings)}\n\nEvidence:\n{_json_dump(evidence)}"
        )
        assistant_payload = {
            "decision": record.get("decision"),
            "recommendations": record.get("recommendations"),
        }
        return user_assistant(
            self.dataset_type,
            record,
            user_text=user,
            assistant_text=_json_dump(assistant_payload),
        )


class EvaluationFormatter(BaseFormatter):
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
