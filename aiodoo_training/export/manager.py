"""ExportManager — atomic publish, manifest, validation, and index updates."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from aiodoo_training.domain.artifacts import EvaluationReport
from aiodoo_training.domain.enums import ExportType
from aiodoo_training.domain.export_manifest import (
    ARTIFACT_PROTOCOL_VERSION,
    ArtifactBundle,
    ArtifactDescriptor,
    ArtifactIndexEntry,
    ArtifactValidationPolicy,
    ExportManifest,
)
from aiodoo_training.domain.export_session import ExportSession
from aiodoo_training.domain.quality import QualityReport
from aiodoo_training.domain.training_policies import TRAINING_PROTOCOL_VERSION
from aiodoo_training.exceptions import ExportError
from aiodoo_training.export.context import ExportContext
from aiodoo_training.export.fingerprints import (
    compute_export_fingerprint,
    compute_model_card_fingerprint,
    sha256_file,
)
from aiodoo_training.export.index import ArtifactIndex
from aiodoo_training.export.lifecycle import ExportLifecycle
from aiodoo_training.export.model_card import (
    ModelCardBuilder,
    ModelCardConfig,
    TrainingSummary,
)

MANIFEST_FILENAME = "export_manifest.json"
CHECKSUMS_FILENAME = "checksums.sha256"
SCHEMA_VERSION = "1"


class ExportManager:
    """
    Application orchestrator for artifact bundle export.

    Owns atomic tmp→rename publish, manifest/checksum writing, and
    ArtifactIndex updates. Delegates file materialization to bound Exporter.
    """

    def __init__(
        self,
        *,
        lifecycle: ExportLifecycle | None = None,
        card_builder: ModelCardBuilder | None = None,
    ) -> None:
        self._lifecycle = lifecycle or ExportLifecycle()
        self._card_builder = card_builder or ModelCardBuilder()

    def export(self, context: ExportContext) -> tuple[ExportContext, ArtifactBundle]:
        """Run preflight, package into tmp, validate, and atomically publish."""
        session = context.export_session
        tmp_root = context.output_dir / f".tmp-export-{uuid4().hex}"
        ctx = context.with_tmp_dir(tmp_root)

        try:
            session = self._lifecycle.preflight(session)
            ctx = ctx.with_export_session(session)

            self._preflight_checks(ctx)

            session = self._lifecycle.begin_packaging(session)
            ctx = ctx.with_export_session(session)
            tmp_root.mkdir(parents=True, exist_ok=True)

            exporter = ctx.exporter
            if hasattr(exporter, "bind"):
                exporter.bind(ctx)

            raw_artifacts = exporter.export(
                ctx.model,
                ctx.export_spec,
                session.experiment_id,
                session.run_id,
            )
            ctx = ctx.with_artifacts(tuple(raw_artifacts))

            card_json, card_paths = self._write_model_card(ctx, tmp_root)
            descriptors = self._collect_descriptors(ctx, tmp_root, card_paths)
            descriptors = self._ensure_required_roles(descriptors, ctx)

            export_fp = compute_export_fingerprint(
                model_fingerprint=ctx.model_fingerprint,
                adapter_fingerprint=ctx.adapter_fingerprint,
                config_fingerprint=ctx.config_fingerprint,
                evaluation_fingerprint=ctx.evaluation_fingerprint,
                model_card_fingerprint=compute_model_card_fingerprint(card_json),
                artifact_descriptors=descriptors,
                export_types=ctx.export_types,
                artifact_protocol_version=ARTIFACT_PROTOCOL_VERSION,
            )

            manifest, manifest_path, checksums_path = self._write_manifest_and_checksums(
                ctx, tmp_root, descriptors, export_fp
            )

            self._validate_bundle(ctx, tmp_root, manifest)
            self._fsync_tree(tmp_root)

            bundle_name = self._bundle_dir_name(session.experiment_id.value, export_fp)
            final_path = ctx.output_dir / bundle_name
            if final_path.exists():
                shutil.rmtree(final_path)
            os.rename(tmp_root, final_path)

            session = self._lifecycle.publish(
                session.with_bundle(final_path, export_fingerprint=export_fp)
            )
            ctx = ctx.with_export_session(session)

            self._update_index(ctx, final_path, manifest)
            bundle = ArtifactBundle(
                root=final_path,
                manifest=manifest,
            )
            return ctx, bundle
        except Exception as exc:
            session = self._safe_fail(session, str(exc))
            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)
            if isinstance(exc, ExportError):
                raise
            raise ExportError(str(exc)) from exc

    def _preflight_checks(self, ctx: ExportContext) -> None:
        if ctx.require_evaluation and ctx.evaluation_report is None:
            raise ExportError("Export requires evaluation report but none was provided.")

        if ctx.require_pass_for_export and ctx.quality_report is not None:
            if not ctx.quality_report.passed:
                if ctx.validation_policy == ArtifactValidationPolicy.RELAXED:
                    return
                if ctx.validation_policy == ArtifactValidationPolicy.WARN:
                    return
                raise ExportError("Quality gates failed and require_pass_for_export is set.")

        policy = ctx.compatibility_policy
        if policy is None:
            return
        if ARTIFACT_PROTOCOL_VERSION not in policy.accepted_artifact_protocols:
            msg = (
                f"artifact_protocol_version {ARTIFACT_PROTOCOL_VERSION!r} not accepted "
                f"by compatibility policy."
            )
            if ctx.validation_policy == ArtifactValidationPolicy.STRICT:
                raise ExportError(msg)
        for role in policy.required_roles:
            if role not in ctx.export_types and role not in ("manifest", "model_card"):
                if ctx.validation_policy == ArtifactValidationPolicy.STRICT:
                    raise ExportError(f"Required export role {role!r} missing from export_types.")

    def _write_model_card(
        self, ctx: ExportContext, tmp_root: Path
    ) -> tuple[dict[str, object], tuple[Path, ...]]:
        if "model_card" not in ctx.export_types:
            return {}, ()
        experiment = self._card_builder.experiment_summary_from_config(
            ctx.config,
            run_id=ctx.export_session.run_id.value,
            config_fingerprint=ctx.config_fingerprint,
        )
        training = TrainingSummary(
            backend_key="stub",
            adaptation_strategy_key="lora",
            seed=ctx.config.seed,
            max_steps=ctx.config.optimization.max_steps,
        )
        evaluation = self._card_builder.evaluation_summary_from_reports(
            ctx.evaluation_report,
            ctx.quality_report,
        )
        md_path, json_path = self._card_builder.write(
            tmp_root,
            experiment=experiment,
            model_fingerprint=ctx.model_fingerprint,
            adapter_fingerprint=ctx.adapter_fingerprint,
            training=training,
            evaluation=evaluation,
            card_config=ModelCardConfig(),
        )
        card_json = json.loads(json_path.read_text(encoding="utf-8"))
        return card_json, (md_path, json_path)

    def _collect_descriptors(
        self,
        ctx: ExportContext,
        tmp_root: Path,
        card_paths: tuple[Path, ...],
    ) -> tuple[ArtifactDescriptor, ...]:
        descriptors: list[ArtifactDescriptor] = []
        for artifact in ctx.artifacts:
            rel = _relative_path(tmp_root, artifact.path)
            checksum = artifact.checksum or sha256_file(str(artifact.path))
            descriptors.append(
                ArtifactDescriptor(
                    role=artifact.export_type.value,
                    relative_path=rel,
                    checksum=checksum,
                    required=True,
                )
            )
        role_map = {
            "model_card.md": "model_card",
            "model_card.json": "model_card",
        }
        for path in card_paths:
            rel = _relative_path(tmp_root, path)
            descriptors.append(
                ArtifactDescriptor(
                    role=role_map.get(path.name, "model_card"),
                    relative_path=rel,
                    checksum=sha256_file(str(path)),
                    content_type="text/markdown" if path.suffix == ".md" else "application/json",
                    required=False,
                )
            )
        if ctx.evaluation_report is not None:
            eval_dir = tmp_root / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)
            report_path = eval_dir / "report.json"
            report_payload = _evaluation_report_to_dict(ctx.evaluation_report)
            report_path.write_text(
                json.dumps(report_payload, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            descriptors.append(
                ArtifactDescriptor(
                    role="evaluation_report",
                    relative_path=_relative_path(tmp_root, report_path),
                    checksum=sha256_file(str(report_path)),
                    required=False,
                )
            )
        if ctx.quality_report is not None:
            eval_dir = tmp_root / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)
            qr_path = eval_dir / "quality_report.json"
            qr_path.write_text(
                json.dumps(_quality_report_to_dict(ctx.quality_report), sort_keys=True, indent=2)
                + "\n",
                encoding="utf-8",
            )
            descriptors.append(
                ArtifactDescriptor(
                    role="quality_report",
                    relative_path=_relative_path(tmp_root, qr_path),
                    checksum=sha256_file(str(qr_path)),
                    required=False,
                )
            )
        return tuple(descriptors)

    def _ensure_required_roles(
        self,
        descriptors: tuple[ArtifactDescriptor, ...],
        ctx: ExportContext,
    ) -> tuple[ArtifactDescriptor, ...]:
        present = {d.role for d in descriptors}
        required: set[str] = set()
        if ExportType.PEFT_ADAPTER.value in ctx.export_types:
            required.add(ExportType.PEFT_ADAPTER.value)
        missing = required - present
        if missing:
            raise ExportError(f"Exporter did not materialize required roles: {sorted(missing)}")
        return descriptors

    def _build_manifest(
        self,
        ctx: ExportContext,
        descriptors: tuple[ArtifactDescriptor, ...],
        export_fingerprint: str,
    ) -> ExportManifest:
        paths = tuple(sorted(d.relative_path for d in descriptors))
        required = tuple(sorted(d.relative_path for d in descriptors if d.required))
        return ExportManifest(
            schema_version=SCHEMA_VERSION,
            artifact_protocol_version=ARTIFACT_PROTOCOL_VERSION,
            experiment_id=ctx.export_session.experiment_id,
            run_id=ctx.export_session.run_id,
            model_fingerprint=ctx.model_fingerprint,
            adapter_fingerprint=ctx.adapter_fingerprint,
            config_fingerprint=ctx.config_fingerprint,
            evaluation_fingerprint=ctx.evaluation_fingerprint,
            export_backend_key=ctx.exporter_backend_key,
            export_types=tuple(ctx.export_types),
            artifacts=descriptors,
            required_artifacts=required,
            artifact_paths=paths,
            export_fingerprint=export_fingerprint,
            training_protocol_version=TRAINING_PROTOCOL_VERSION,
            created_at=datetime.now(UTC),
            software={
                "python": (
                    f"{sys.version_info.major}."
                    f"{sys.version_info.minor}."
                    f"{sys.version_info.micro}"
                ),
                "aiodoo-training": "phase4",
            },
        )

    def _write_checksums(
        self,
        checksums_path: Path,
        tmp_root: Path,
        descriptors: tuple[ArtifactDescriptor, ...],
    ) -> None:
        lines: list[str] = []
        for descriptor in sorted(descriptors, key=lambda d: d.relative_path):
            file_path = tmp_root / descriptor.relative_path
            if file_path.is_file():
                lines.append(f"{descriptor.checksum}  {descriptor.relative_path}")
        checksums_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _write_manifest_and_checksums(
        self,
        ctx: ExportContext,
        tmp_root: Path,
        content_descriptors: tuple[ArtifactDescriptor, ...],
        export_fp: str,
    ) -> tuple[ExportManifest, Path, Path]:
        """Write checksums + manifest with stable content checksums."""
        checksums_path = tmp_root / CHECKSUMS_FILENAME
        manifest_path = tmp_root / MANIFEST_FILENAME

        self._write_checksums(checksums_path, tmp_root, content_descriptors)
        checksums_descriptor = ArtifactDescriptor(
            role="checksums",
            relative_path=CHECKSUMS_FILENAME,
            checksum=sha256_file(str(checksums_path)),
            required=True,
        )

        manifest = self._build_manifest(
            ctx, content_descriptors + (checksums_descriptor,), export_fp
        )
        manifest_path.write_text(
            json.dumps(_manifest_to_dict(manifest), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest_descriptor = ArtifactDescriptor(
            role=ExportType.MANIFEST.value,
            relative_path=MANIFEST_FILENAME,
            checksum=sha256_file(str(manifest_path)),
            required=True,
        )
        manifest = self._build_manifest(
            ctx,
            content_descriptors + (checksums_descriptor, manifest_descriptor),
            export_fp,
        )
        manifest_path.write_text(
            json.dumps(_manifest_to_dict(manifest), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest, manifest_path, checksums_path

    def _validate_bundle(
        self,
        ctx: ExportContext,
        tmp_root: Path,
        manifest: ExportManifest,
    ) -> None:
        policy = ctx.validation_policy
        for descriptor in manifest.artifacts:
            if not descriptor.required:
                continue
            file_path = tmp_root / descriptor.relative_path
            if not file_path.is_file():
                raise ExportError(f"Required artifact missing: {descriptor.relative_path}")

        if policy in {ArtifactValidationPolicy.STRICT, ArtifactValidationPolicy.WARN}:
            for descriptor in manifest.artifacts:
                file_path = tmp_root / descriptor.relative_path
                if not file_path.is_file():
                    continue
                actual = sha256_file(str(file_path))
                if actual != descriptor.checksum:
                    msg = f"Checksum mismatch for {descriptor.relative_path}"
                    if descriptor.relative_path in {MANIFEST_FILENAME, CHECKSUMS_FILENAME}:
                        continue
                    if policy == ArtifactValidationPolicy.STRICT:
                        raise ExportError(msg)

        if policy == ArtifactValidationPolicy.STRICT:
            if manifest.model_fingerprint != ctx.model_fingerprint:
                raise ExportError("Manifest model_fingerprint does not match context.")
            if manifest.adapter_fingerprint != ctx.adapter_fingerprint:
                raise ExportError("Manifest adapter_fingerprint does not match context.")

    def _update_index(
        self,
        ctx: ExportContext,
        bundle_path: Path,
        manifest: ExportManifest,
    ) -> None:
        index = ArtifactIndex.load(ctx.output_dir)
        rel_bundle = _relative_path(ctx.output_dir, bundle_path)
        roles = tuple(sorted({d.role for d in manifest.artifacts}))
        entry = ArtifactIndexEntry(
            bundle_path=rel_bundle,
            experiment_id=manifest.experiment_id,
            run_id=manifest.run_id,
            export_fingerprint=manifest.export_fingerprint,
            artifact_protocol_version=manifest.artifact_protocol_version,
            export_types=manifest.export_types,
            roles=roles,
            created_at=manifest.created_at,
        )
        index = index.upsert(entry)
        index.save()

    def _fsync_tree(self, root: Path) -> None:
        for path in root.rglob("*"):
            if path.is_file():
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _append_descriptor(
        self,
        descriptors: tuple[ArtifactDescriptor, ...],
        *,
        role: str,
        relative_path: str,
        path: Path,
        required: bool,
    ) -> tuple[ArtifactDescriptor, ...]:
        desc = ArtifactDescriptor(
            role=role,
            relative_path=relative_path,
            checksum=sha256_file(str(path)),
            required=required,
        )
        return descriptors + (desc,)

    def _bundle_dir_name(self, experiment_id: str, export_fingerprint: str) -> str:
        short_fp = export_fingerprint[:12]
        safe_exp = experiment_id.replace("/", "_").replace(" ", "_")
        return f"bundle-{safe_exp}-{short_fp}"

    def _safe_fail(self, session: ExportSession, message: str) -> ExportSession:
        try:
            return self._lifecycle.fail(session, message=message)
        except Exception:  # noqa: BLE001
            return session.with_status(session.status, message=message)


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _manifest_to_dict(manifest: ExportManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "artifact_protocol_version": manifest.artifact_protocol_version,
        "training_protocol_version": manifest.training_protocol_version,
        "experiment_id": manifest.experiment_id.value,
        "run_id": manifest.run_id.value,
        "model_fingerprint": manifest.model_fingerprint,
        "adapter_fingerprint": manifest.adapter_fingerprint,
        "config_fingerprint": manifest.config_fingerprint,
        "evaluation_fingerprint": manifest.evaluation_fingerprint,
        "export_backend_key": manifest.export_backend_key,
        "export_types": list(manifest.export_types),
        "artifacts": [
            {
                "role": d.role,
                "relative_path": d.relative_path,
                "checksum": d.checksum,
                "content_type": d.content_type,
                "required": d.required,
            }
            for d in manifest.artifacts
        ],
        "required_artifacts": list(manifest.required_artifacts),
        "artifact_paths": list(manifest.artifact_paths),
        "export_fingerprint": manifest.export_fingerprint,
        "created_at": manifest.created_at.isoformat() if manifest.created_at else None,
        "software": dict(manifest.software),
    }


def _evaluation_report_to_dict(report: EvaluationReport) -> dict[str, object]:
    return {
        "experiment_id": report.experiment_id.value,
        "run_id": report.run_id.value,
        "passed": report.passed,
        "details": report.details,
        "metrics": [
            {
                "name": m.name,
                "value": m.value,
                "step": m.step,
            }
            for m in report.metrics
        ],
    }


def _quality_report_to_dict(report: QualityReport) -> dict[str, object]:
    return {
        "passed": report.passed,
        "details": report.details,
        "failures": [
            {
                "metric_key": f.metric_key,
                "message": f.message,
                "severity": f.severity,
                "observed": f.observed,
                "expected": f.expected,
            }
            for f in report.failures
        ],
        "warnings": [
            {
                "metric_key": w.metric_key,
                "message": w.message,
                "severity": w.severity,
            }
            for w in report.warnings
        ],
    }
