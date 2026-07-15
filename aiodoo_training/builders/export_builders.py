"""Phase 4 ExportBuilder / ExportContextBuilder."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from aiodoo_training.domain.config import ExperimentConfig, ExportSpec
from aiodoo_training.domain.export_manifest import (
    ArtifactCompatibilityPolicy,
    ArtifactValidationPolicy,
)
from aiodoo_training.domain.export_session import ExportSession
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.exceptions import BuilderError

if TYPE_CHECKING:
    from aiodoo_training.export.context import ExportContext


class ExportBuilder:
    """Assembles export profile options into ExportSpec / policy knobs."""

    def __init__(self) -> None:
        self._backend_key = "stub"
        self._export_types: tuple[str, ...] = (
            "peft_adapter",
            "tokenizer",
            "manifest",
            "model_card",
            "bundle",
        )
        self._output_dir: Path = Path("artifacts/export")
        self._require_evaluation = False
        self._require_pass_for_export = False
        self._validation_policy = ArtifactValidationPolicy.STRICT

    def with_backend(self, key: str) -> ExportBuilder:
        self._backend_key = key
        return self

    def with_export_types(self, *types: str) -> ExportBuilder:
        self._export_types = tuple(types)
        return self

    def with_output_dir(self, path: Path) -> ExportBuilder:
        self._output_dir = path
        return self

    def with_require_evaluation(self, required: bool) -> ExportBuilder:
        self._require_evaluation = required
        return self

    def with_validation_policy(self, policy: ArtifactValidationPolicy) -> ExportBuilder:
        self._validation_policy = policy
        return self

    def build_spec(self) -> ExportSpec:
        return ExportSpec(output_dir=self._output_dir, export_types=self._export_types)

    @property
    def backend_key(self) -> str:
        return self._backend_key

    @property
    def require_evaluation(self) -> bool:
        return self._require_evaluation

    @property
    def require_pass_for_export(self) -> bool:
        return self._require_pass_for_export

    @property
    def validation_policy(self) -> ArtifactValidationPolicy:
        return self._validation_policy

    @property
    def export_types(self) -> tuple[str, ...]:
        return self._export_types


class ExportContextBuilder:
    """Builds a resolved :class:`ExportContext` from collaborator pieces."""

    def __init__(self) -> None:
        self._pieces: dict[str, object] = {}

    def with_config(self, config: ExperimentConfig) -> ExportContextBuilder:
        self._pieces["config"] = config
        return self

    def with_piece(self, key: str, value: object) -> ExportContextBuilder:
        self._pieces[key] = value
        return self

    def build(self, config: ExperimentConfig | None = None) -> ExportContext:
        cfg = config if config is not None else self._pieces.get("config")
        if not isinstance(cfg, ExperimentConfig):
            raise BuilderError("ExportContextBuilder requires ExperimentConfig.")

        required = ("execution", "model", "exporter", "export_session", "output_dir")
        missing = [key for key in required if key not in self._pieces]
        if missing:
            raise BuilderError(
                "ExportContextBuilder missing required pieces: " + ", ".join(missing)
            )

        export_spec = self._pieces.get("export_spec")
        if not isinstance(export_spec, ExportSpec):
            export_spec = cfg.export

        output_dir = Path(self._pieces["output_dir"])  # type: ignore[arg-type]
        tmp_dir = self._pieces.get("tmp_dir")
        if tmp_dir is None:
            tmp_dir = output_dir / ".tmp-export-pending"
        else:
            tmp_dir = Path(tmp_dir)  # type: ignore[arg-type]

        validation_policy = self._pieces.get("validation_policy")
        if not isinstance(validation_policy, ArtifactValidationPolicy):
            validation_policy = ArtifactValidationPolicy.STRICT

        compatibility = self._pieces.get("compatibility_policy")
        if compatibility is not None and not isinstance(compatibility, ArtifactCompatibilityPolicy):
            raise BuilderError("compatibility_policy must be ArtifactCompatibilityPolicy")

        export_types_obj = self._pieces.get("export_types")
        if export_types_obj is None:
            export_types = export_spec.export_types
        elif isinstance(export_types_obj, tuple):
            export_types = export_types_obj
        else:
            export_types = tuple(export_types_obj)  # type: ignore[arg-type]

        bind_extra_obj = self._pieces.get("bind_extra") or {}
        if not isinstance(bind_extra_obj, dict):
            raise BuilderError("bind_extra must be a mapping when provided.")
        bind_extra = {str(k): v for k, v in bind_extra_obj.items()}

        from aiodoo_training.export.context import ExportContext

        return ExportContext(
            config=cfg,
            export_spec=export_spec,
            model=self._pieces["model"],  # type: ignore[arg-type]
            execution=self._pieces["execution"],  # type: ignore[arg-type]
            export_session=self._pieces["export_session"],  # type: ignore[arg-type]
            exporter=self._pieces["exporter"],  # type: ignore[arg-type]
            output_dir=output_dir,
            tmp_dir=tmp_dir,
            exporter_backend_key=str(self._pieces.get("exporter_backend_key") or "stub"),
            model_fingerprint=str(self._pieces.get("model_fingerprint") or ""),
            adapter_fingerprint=str(self._pieces.get("adapter_fingerprint") or ""),
            config_fingerprint=str(self._pieces.get("config_fingerprint") or ""),
            evaluation_fingerprint=(
                str(self._pieces["evaluation_fingerprint"])
                if self._pieces.get("evaluation_fingerprint") is not None
                else None
            ),
            evaluation_report=self._pieces.get("evaluation_report"),  # type: ignore[arg-type]
            quality_report=self._pieces.get("quality_report"),  # type: ignore[arg-type]
            validation_policy=validation_policy,
            compatibility_policy=compatibility,
            require_evaluation=bool(self._pieces.get("require_evaluation") or False),
            require_pass_for_export=bool(self._pieces.get("require_pass_for_export") or False),
            export_types=export_types,
            tracker=self._pieces.get("tracker"),  # type: ignore[arg-type]
            artifacts=tuple(self._pieces.get("artifacts") or ()),  # type: ignore[arg-type]
            bind_extra=bind_extra,
        )


def make_export_session(
    *,
    experiment_id: ExperimentId,
    run_id: RunId,
    model_fingerprint: str = "",
    adapter_fingerprint: str = "",
    config_fingerprint: str = "",
) -> ExportSession:
    """Create a PENDING ExportSession."""
    now = datetime.now(UTC)
    return ExportSession(
        session_id=f"export-{uuid4().hex[:12]}",
        experiment_id=experiment_id,
        run_id=run_id,
        model_fingerprint=model_fingerprint,
        adapter_fingerprint=adapter_fingerprint,
        config_fingerprint=config_fingerprint,
        created_at=now,
        updated_at=now,
    )
