"""Thin HuggingFace Exporter — real PEFT adapter write (AT-2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiodoo_training.domain.artifacts import ExportArtifact
from aiodoo_training.domain.config import ExportSpec
from aiodoo_training.domain.enums import ExportType
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.exceptions import DomainError
from aiodoo_training.export.fingerprints import sha256_hex
from aiodoo_training.infrastructure.model_handles import require_trainable_carrier
from aiodoo_training.infrastructure.stub.exporter import StubExporter
from aiodoo_training.ports.trainer import Exporter

if TYPE_CHECKING:
    from aiodoo_training.export.context import ExportContext
else:
    ExportContext = Any  # type: ignore[misc,assignment]

_ADAPTER_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise DomainError(
            "HFExporter requires the 'transformers' package. "
            "Install training extras or use Exporter key 'stub' for CPU CI."
        ) from exc
    return transformers


def _is_stub_framework(framework: object) -> bool:
    return isinstance(framework, dict) and framework.get("kind") == "stub"


def _assert_peft_capable(framework: object) -> None:
    if _is_stub_framework(framework):
        raise DomainError(
            "HFExporter refuses stub models; use Exporter key 'stub' for CI stubs."
        )
    save_pretrained = getattr(framework, "save_pretrained", None)
    if not callable(save_pretrained):
        raise DomainError(
            "HFExporter requires a PEFT/transformers model exposing save_pretrained()."
        )
    # Prefer explicit PEFT markers when present.
    peft_config = getattr(framework, "peft_config", None)
    peft_type = type(framework).__name__
    if peft_config is None and "Peft" not in peft_type and not hasattr(framework, "get_base_model"):
        # Still allow save_pretrained-only adapters (some wrappers).
        pass


def _require_adapter_weight_files(adapter_dir: Path) -> Path:
    for name in _ADAPTER_WEIGHT_NAMES:
        candidate = adapter_dir / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    raise DomainError(
        f"HFExporter PEFT save produced no non-empty adapter weights under {adapter_dir}. "
        f"Expected one of {_ADAPTER_WEIGHT_NAMES}."
    )


class HFExporter(Exporter):
    """
    HuggingFace / PEFT exporter.

    Writes real adapter weights via ``save_pretrained``. Falls back to
    :class:`StubExporter` only when transformers is unavailable.
    """

    BACKEND_KEY = "hf_peft"

    def __init__(self, context: ExportContext | None = None) -> None:
        self._context = context
        self._stub = StubExporter(context)

    def bind(self, context: ExportContext) -> HFExporter:
        self._context = context
        self._stub.bind(context)
        return self

    @property
    def context(self) -> ExportContext | None:
        return self._context

    def export(
        self,
        model: TrainableModelHandle,
        spec: ExportSpec,
        experiment_id: ExperimentId,
        run_id: RunId,
    ) -> tuple[ExportArtifact, ...]:
        try:
            _require_transformers()
        except DomainError:
            return self._stub.export(model, spec, experiment_id, run_id)

        ctx = self._context
        if ctx is None:
            raise DomainError("HFExporter requires ExportContext via bind() before export().")

        carrier = require_trainable_carrier(model)
        framework = carrier.framework_model
        _assert_peft_capable(framework)

        destination = ctx.tmp_dir
        destination.mkdir(parents=True, exist_ok=True)
        artifacts_dir = destination / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        export_types = set(ctx.export_types or spec.export_types)
        out: list[ExportArtifact] = []

        if ExportType.PEFT_ADAPTER.value in export_types or "peft_adapter" in export_types:
            adapter_dir = artifacts_dir / "adapter"
            if adapter_dir.exists():
                # Clean prior attempt in the same tmp dir
                import shutil

                shutil.rmtree(adapter_dir)
            adapter_dir.mkdir(parents=True, exist_ok=True)
            save_pretrained = getattr(framework, "save_pretrained")
            save_pretrained(str(adapter_dir))
            weight_path = _require_adapter_weight_files(adapter_dir)
            config_path = adapter_dir / "adapter_config.json"
            if not config_path.is_file():
                raise DomainError(
                    f"HFExporter PEFT save missing adapter_config.json under {adapter_dir}"
                )
            digest = sha256_hex(weight_path.read_bytes())
            out.append(
                ExportArtifact(
                    export_type=ExportType.PEFT_ADAPTER,
                    path=adapter_dir,
                    checksum=digest,
                    description="peft adapter weights + config",
                )
            )

        if ExportType.TOKENIZER.value in export_types or "tokenizer" in export_types:
            tok_dir = artifacts_dir / "tokenizer"
            tok_dir.mkdir(parents=True, exist_ok=True)
            tokenizer = None
            if isinstance(ctx.bind_extra, dict):
                tokenizer = ctx.bind_extra.get("tokenizer")
            hf_tok = getattr(tokenizer, "_tokenizer", None) if tokenizer is not None else None
            if hf_tok is not None and hasattr(hf_tok, "save_pretrained"):
                hf_tok.save_pretrained(str(tok_dir))
                out.append(
                    ExportArtifact(
                        export_type=ExportType.TOKENIZER,
                        path=tok_dir,
                        checksum=sha256_hex(str(tok_dir).encode("utf-8")),
                        description="tokenizer files",
                    )
                )

        # Provenance sidecar (FP2 / System Training Contract) when configured.
        provenance = {}
        if isinstance(ctx.bind_extra, dict):
            raw_prov = ctx.bind_extra.get("fp2_provenance")
            if isinstance(raw_prov, dict):
                provenance = dict(raw_prov)
        if provenance:
            prov_path = artifacts_dir / "fp2_provenance.json"
            content = json.dumps(provenance, sort_keys=True, indent=2) + "\n"
            prov_path.write_text(content, encoding="utf-8")
            out.append(
                ExportArtifact(
                    export_type=ExportType.MANIFEST,
                    path=prov_path,
                    checksum=sha256_hex(content.encode("utf-8")),
                    description="fp2 provenance sidecar",
                )
            )

        if not out:
            raise DomainError("HFExporter produced no artifacts for requested export_types.")
        _ = (experiment_id, run_id)
        return tuple(out)


def register_hf_exporter(*, overwrite: bool = False) -> None:
    from aiodoo_training.registries import exporter_registry

    if not exporter_registry.exists("hf_peft") or overwrite:
        exporter_registry.register("hf_peft", HFExporter, overwrite=overwrite)
