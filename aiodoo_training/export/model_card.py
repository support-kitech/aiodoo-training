"""Model card builder — deterministic markdown + JSON sidecars."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiodoo_training.domain.artifacts import EvaluationReport
from aiodoo_training.domain.config import ExperimentConfig
from aiodoo_training.domain.quality import QualityReport
from aiodoo_training.export.fingerprints import compute_model_card_fingerprint


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    """Portable experiment identity + config digest."""

    experiment_id: str
    run_id: str
    name: str
    config_fingerprint: str


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Portable training provenance summary."""

    backend_key: str
    adaptation_strategy_key: str
    seed: int
    max_steps: int | None = None


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Portable evaluation metrics + gate outcome."""

    passed: bool
    metrics: dict[str, float]
    gate_passed: bool | None = None


@dataclass(frozen=True, slots=True)
class ModelCardConfig:
    """Declarative model card fields from config."""

    license: str = "Apache-2.0"
    limitations: str = ""
    include_training_summary: bool = True
    include_evaluation_summary: bool = True


class ModelCardBuilder:
    """Assemble model_card.md + model_card.json from portable summaries."""

    def build_json(
        self,
        *,
        experiment: ExperimentSummary,
        model_fingerprint: str,
        adapter_fingerprint: str,
        training: TrainingSummary | None = None,
        evaluation: EvaluationSummary | None = None,
        card_config: ModelCardConfig | None = None,
    ) -> dict[str, Any]:
        cfg = card_config or ModelCardConfig()
        payload: dict[str, Any] = {
            "schema_version": "1",
            "experiment_id": experiment.experiment_id,
            "run_id": experiment.run_id,
            "name": experiment.name,
            "config_fingerprint": experiment.config_fingerprint,
            "model_fingerprint": model_fingerprint,
            "adapter_fingerprint": adapter_fingerprint,
            "license": cfg.license,
            "limitations": cfg.limitations,
        }
        if cfg.include_training_summary and training is not None:
            payload["training"] = {
                "backend_key": training.backend_key,
                "adaptation_strategy_key": training.adaptation_strategy_key,
                "seed": training.seed,
                "max_steps": training.max_steps,
            }
        if cfg.include_evaluation_summary and evaluation is not None:
            payload["evaluation"] = {
                "passed": evaluation.passed,
                "metrics": evaluation.metrics,
                "gate_passed": evaluation.gate_passed,
            }
        payload["model_card_fingerprint"] = compute_model_card_fingerprint(payload)
        return payload

    def build_markdown(self, card_json: dict[str, Any]) -> str:
        lines = [
            f"# Model Card: {card_json.get('name', 'unknown')}",
            "",
            f"- **Experiment ID:** {card_json.get('experiment_id')}",
            f"- **Run ID:** {card_json.get('run_id')}",
            f"- **Model fingerprint:** `{card_json.get('model_fingerprint')}`",
            f"- **Adapter fingerprint:** `{card_json.get('adapter_fingerprint')}`",
            f"- **License:** {card_json.get('license')}",
            "",
        ]
        limitations = card_json.get("limitations")
        if limitations:
            lines.extend(["## Limitations", "", str(limitations), ""])
        training = card_json.get("training")
        if isinstance(training, dict):
            lines.extend(
                [
                    "## Training",
                    "",
                    f"- Backend: {training.get('backend_key')}",
                    f"- Adaptation: {training.get('adaptation_strategy_key')}",
                    f"- Seed: {training.get('seed')}",
                    "",
                ]
            )
        evaluation = card_json.get("evaluation")
        if isinstance(evaluation, dict):
            lines.extend(["## Evaluation", ""])
            metrics = evaluation.get("metrics") or {}
            if isinstance(metrics, dict):
                for key in sorted(metrics):
                    lines.append(f"- {key}: {metrics[key]}")
            lines.append("")
        return "\n".join(lines)

    def write(
        self,
        destination: Path,
        *,
        experiment: ExperimentSummary,
        model_fingerprint: str,
        adapter_fingerprint: str,
        training: TrainingSummary | None = None,
        evaluation: EvaluationSummary | None = None,
        card_config: ModelCardConfig | None = None,
    ) -> tuple[Path, Path]:
        """Write model_card.json and model_card.md under destination."""
        destination.mkdir(parents=True, exist_ok=True)
        card_json = self.build_json(
            experiment=experiment,
            model_fingerprint=model_fingerprint,
            adapter_fingerprint=adapter_fingerprint,
            training=training,
            evaluation=evaluation,
            card_config=card_config,
        )
        json_path = destination / "model_card.json"
        md_path = destination / "model_card.md"
        json_path.write_text(
            json.dumps(card_json, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(self.build_markdown(card_json), encoding="utf-8")
        return md_path, json_path

    @staticmethod
    def evaluation_summary_from_reports(
        report: EvaluationReport | None,
        quality: QualityReport | None,
    ) -> EvaluationSummary | None:
        if report is None:
            return None
        metrics = {snap.name: snap.value for snap in report.metrics}
        gate_passed = quality.passed if quality is not None else None
        return EvaluationSummary(
            passed=report.passed,
            metrics=metrics,
            gate_passed=gate_passed,
        )

    @staticmethod
    def experiment_summary_from_config(
        config: ExperimentConfig,
        *,
        run_id: str,
        config_fingerprint: str,
    ) -> ExperimentSummary:
        experiment_id = config.experiment_id.value if config.experiment_id else config.name
        return ExperimentSummary(
            experiment_id=experiment_id,
            run_id=run_id,
            name=config.name,
            config_fingerprint=config_fingerprint,
        )
