"""CPU stub Exporter — writes deterministic adapter/tokenizer stubs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from aiodoo_training.domain.artifacts import ExportArtifact
from aiodoo_training.domain.config import ExportSpec
from aiodoo_training.domain.enums import ExportType
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.export.fingerprints import sha256_hex
from aiodoo_training.infrastructure.model_handles import require_trainable_carrier
from aiodoo_training.ports.trainer import Exporter

if TYPE_CHECKING:
    from aiodoo_training.export.context import ExportContext
else:
    ExportContext = Any  # type: ignore[misc,assignment]


def _stable_adapter_payload(
    *,
    model_fingerprint: str,
    adapter_fingerprint: str,
    experiment_id: str,
    run_id: str,
) -> dict[str, object]:
    return {
        "format": "peft_adapter_stub_v1",
        "experiment_id": experiment_id,
        "run_id": run_id,
        "model_fingerprint": model_fingerprint,
        "adapter_fingerprint": adapter_fingerprint,
        "weights": [0.01, 0.02, 0.03, 0.04],
        "config": {"rank": 8, "alpha": 16, "target_modules": ["q_proj"]},
    }


def _stable_tokenizer_payload(*, model_fingerprint: str) -> dict[str, object]:
    return {
        "format": "tokenizer_stub_v1",
        "model_fingerprint": model_fingerprint,
        "vocab_size": 32000,
        "special_tokens": {"pad": "<pad>", "eos": "</s>", "bos": "<s>"},
    }


class StubExporter(Exporter):
    """
    Deterministic CPU exporter for CI golden tests.

    Writes relative to bound :class:`ExportContext`.tmp_dir — ExportManager
    owns atomicity. Never evaluates.
    """

    BACKEND_KEY = "stub"

    def __init__(self, context: ExportContext | None = None) -> None:
        self._context = context

    def bind(self, context: ExportContext) -> StubExporter:
        self._context = context
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
        _ = (model, spec)
        ctx = self._context
        if ctx is None:
            raise RuntimeError("StubExporter requires ExportContext via bind() before export().")

        destination = ctx.tmp_dir
        destination.mkdir(parents=True, exist_ok=True)
        artifacts_dir = destination / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        carrier = require_trainable_carrier(model)
        model_fp = ctx.model_fingerprint or "unknown-model"
        adapter_fp = ctx.adapter_fingerprint or "unknown-adapter"
        export_types = set(ctx.export_types or spec.export_types)

        out: list[ExportArtifact] = []

        if ExportType.PEFT_ADAPTER.value in export_types or "peft_adapter" in export_types:
            adapter_dir = artifacts_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            payload = _stable_adapter_payload(
                model_fingerprint=model_fp,
                adapter_fingerprint=adapter_fp,
                experiment_id=experiment_id.value,
                run_id=run_id.value,
            )
            adapter_path = adapter_dir / "adapter_stub.json"
            content = json.dumps(payload, sort_keys=True, indent=2) + "\n"
            adapter_path.write_text(content, encoding="utf-8")
            out.append(
                ExportArtifact(
                    export_type=ExportType.PEFT_ADAPTER,
                    path=adapter_path,
                    checksum=sha256_hex(content.encode("utf-8")),
                    description="stub peft adapter weights",
                )
            )

        if ExportType.TOKENIZER.value in export_types or "tokenizer" in export_types:
            tok_dir = artifacts_dir / "tokenizer"
            tok_dir.mkdir(parents=True, exist_ok=True)
            tok_payload = _stable_tokenizer_payload(model_fingerprint=model_fp)
            tok_path = tok_dir / "tokenizer_stub.json"
            tok_content = json.dumps(tok_payload, sort_keys=True, indent=2) + "\n"
            tok_path.write_text(tok_content, encoding="utf-8")
            out.append(
                ExportArtifact(
                    export_type=ExportType.TOKENIZER,
                    path=tok_path,
                    checksum=sha256_hex(tok_content.encode("utf-8")),
                    description="stub tokenizer",
                )
            )

        if ExportType.MERGED_WEIGHTS.value in export_types:
            merged_dir = artifacts_dir / "merged"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_payload = {
                "format": "merged_weights_stub_v1",
                "model_fingerprint": model_fp,
                "adapter_fingerprint": adapter_fp,
            }
            merged_path = merged_dir / "merged_stub.json"
            merged_content = json.dumps(merged_payload, sort_keys=True, indent=2) + "\n"
            merged_path.write_text(merged_content, encoding="utf-8")
            out.append(
                ExportArtifact(
                    export_type=ExportType.MERGED_WEIGHTS,
                    path=merged_path,
                    checksum=sha256_hex(merged_content.encode("utf-8")),
                    description="stub merged weights",
                )
            )

        _ = carrier
        return tuple(out)


def register_default_exporters(*, overwrite: bool = False) -> None:
    """Register ``stub`` and lazy ``hf_peft`` exporters."""
    from aiodoo_training.infrastructure.huggingface.exporter import register_hf_exporter
    from aiodoo_training.registries import exporter_registry

    if not exporter_registry.exists("stub") or overwrite:
        exporter_registry.register("stub", StubExporter, overwrite=overwrite)
    register_hf_exporter(overwrite=overwrite)
