"""SchedulePlanner — sole Phase 5 orchestration owner."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from aiodoo_training.curriculum import (
    CurriculumContext,
    CurriculumLifecycle,
    fingerprint_stages,
)
from aiodoo_training.domain.config import CurriculumSpec, PackingSpec
from aiodoo_training.domain.curriculum_session import CurriculumSession, CurriculumStatistics
from aiodoo_training.domain.enums import (
    CurriculumMode,
    CurriculumStatus,
    PackingMode,
    PackingStatus,
)
from aiodoo_training.domain.examples import TokenBatch, TrainingExample
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.packing_policies import PackingPolicy, SamplingSpec
from aiodoo_training.domain.packing_session import PackingSession, PackingStatistics
from aiodoo_training.packing.context import PackingContext
from aiodoo_training.packing.lifecycle import PackingLifecycle
from aiodoo_training.packing.token_rows import resolve_token_rows
from aiodoo_training.ports.packing import CurriculumStrategy, PackingStrategy, SamplingStrategy


def _sha(*parts: str) -> str:
    material = "|".join(parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _canonical(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True, slots=True)
class SchedulePlan:
    """Completed curriculum + sampling + packing plan."""

    ordered_examples: tuple[TrainingExample, ...]
    curriculum_stages: tuple[tuple[TrainingExample, ...], ...]
    token_batches: tuple[TokenBatch, ...]
    packing_statistics: PackingStatistics
    curriculum_statistics: CurriculumStatistics
    packing_fingerprint: str
    curriculum_fingerprint: str
    sampling_fingerprint: str
    packing_session: PackingSession
    curriculum_session: CurriculumSession
    packing_context: PackingContext
    curriculum_context: CurriculumContext


class SchedulePlanner:
    """
    Sole Phase 5 orchestrator: curriculum → sampling → packing → statistics.

    Idempotent: calling ensure_order twice with the same inputs yields equal
    fingerprints and statistics (excluding session ids / timestamps).
    """

    def __init__(
        self,
        *,
        packing_lifecycle: PackingLifecycle | None = None,
        curriculum_lifecycle: CurriculumLifecycle | None = None,
    ) -> None:
        self._packing_life = packing_lifecycle or PackingLifecycle()
        self._curriculum_life = curriculum_lifecycle or CurriculumLifecycle()

    def ensure_order(
        self,
        examples: Sequence[TrainingExample],
        *,
        curriculum: CurriculumStrategy,
        sampling: SamplingStrategy,
        packing: PackingStrategy,
        curriculum_spec: CurriculumSpec,
        packing_spec: PackingSpec,
        sampling_spec: SamplingSpec,
        packing_policy: PackingPolicy,
        experiment_id: ExperimentId,
        run_id: RunId,
        seed: int = 42,
        packing_session: PackingSession | None = None,
        curriculum_session: CurriculumSession | None = None,
    ) -> SchedulePlan:
        now = datetime.now(UTC)
        examples_t = tuple(examples)

        c_session = curriculum_session or CurriculumSession(
            session_id=f"cur-{uuid4().hex[:12]}",
            experiment_id=experiment_id,
            run_id=run_id,
            created_at=now,
            updated_at=now,
        )
        p_session = packing_session or PackingSession(
            session_id=f"pack-{uuid4().hex[:12]}",
            experiment_id=experiment_id,
            run_id=run_id,
            created_at=now,
            updated_at=now,
        )

        c_ctx = CurriculumContext(
            curriculum_session=c_session,
            curriculum_spec=curriculum_spec,
            seed=seed,
            backend_key=getattr(curriculum, "BACKEND_KEY", "none"),
        )
        if hasattr(curriculum, "bind"):
            curriculum.bind(c_ctx)

        # Curriculum
        if curriculum_spec.mode == CurriculumMode.NONE and getattr(
            curriculum, "BACKEND_KEY", ""
        ) in {"", "none"}:
            if c_session.status == CurriculumStatus.PENDING:
                c_session = self._curriculum_life.skip(c_session, message="curriculum none")
            stages: tuple[tuple[TrainingExample, ...], ...] = (examples_t,)
        else:
            if c_session.status == CurriculumStatus.PENDING:
                c_session = self._curriculum_life.begin(c_session)
            stages = tuple(tuple(s) for s in curriculum.plan(examples_t, curriculum_spec))
            if c_session.status == CurriculumStatus.PLANNING:
                c_session = self._curriculum_life.ready(c_session)
            c_session = c_session.with_stage(
                stage_index=0,
                stage_count=len(stages),
                examples_in_stage=len(stages[0]) if stages else 0,
            )

        # Sampling per stage then flatten
        if hasattr(sampling, "bind"):
            sampling.bind({"seed": seed})
        sampled_stages: list[tuple[TrainingExample, ...]] = []
        for stage in stages:
            sampled_stages.append(tuple(sampling.sample(stage, sampling_spec)))
        ordered = tuple(ex for stage in sampled_stages for ex in stage)

        curriculum_fp = fingerprint_stages(sampled_stages)
        c_session = c_session.with_fingerprint(curriculum_fp)
        sampling_fp = _sha(
            sampling_spec.backend_key,
            str(sampling_spec.seed),
            str(sampling_spec.temperature),
            ",".join(e.example_id for e in ordered),
        )

        cur_stats = CurriculumStatistics(
            curriculum_fingerprint=curriculum_fp,
            backend_key=getattr(curriculum, "BACKEND_KEY", "none"),
            stage_count=len(sampled_stages),
            examples_total=sum(len(s) for s in sampled_stages),
            examples_per_stage=tuple(len(s) for s in sampled_stages),
            stage_names=tuple(curriculum_spec.stages),
            weight_per_stage=(),
        )

        # Packing
        token_rows = resolve_token_rows(
            ordered, max_length=packing_policy.max_sequence_length
        )
        p_ctx = PackingContext(
            examples=ordered,
            packing_session=p_session,
            packing_spec=packing_spec,
            packing_policy=packing_policy,
            token_rows=token_rows,
            seed=seed,
        )
        if hasattr(packing, "bind"):
            packing.bind(p_ctx)

        if packing_policy.mode == PackingMode.NONE and packing_policy.backend_key == "none":
            # Still produce passthrough batches via none strategy
            pass

        if p_session.status == PackingStatus.PENDING:
            if packing_policy.mode == PackingMode.NONE and packing_policy.backend_key == "none":
                # Keep planning path consistent — begin→ready still for stats.
                p_session = self._packing_life.begin(p_session, message="packing")
            else:
                p_session = self._packing_life.begin(p_session)

        batches = tuple(packing.pack(ordered, packing_spec))
        # Re-read context for overflow counters after pack
        bound = getattr(packing, "context", None) or getattr(packing, "_context", None)
        if isinstance(bound, PackingContext):
            p_ctx = bound

        pack_stats = self._build_packing_statistics(
            batches,
            packing_policy=packing_policy,
            examples_input=len(ordered),
            backend_key=getattr(packing, "BACKEND_KEY", packing_policy.backend_key),
            overflow_deferred=p_ctx.overflow_deferred,
            overflow_truncated=p_ctx.overflow_truncated,
        )
        packing_fp = pack_stats.packing_fingerprint
        p_session = p_session.with_fingerprint(packing_fp)
        if p_session.status == PackingStatus.PLANNING:
            p_session = self._packing_life.ready(p_session)
        p_ctx = p_ctx.with_session(p_session).with_statistics(pack_stats)
        c_ctx = c_ctx.with_session(c_session)

        return SchedulePlan(
            ordered_examples=ordered,
            curriculum_stages=tuple(sampled_stages),
            token_batches=batches,
            packing_statistics=pack_stats,
            curriculum_statistics=cur_stats,
            packing_fingerprint=packing_fp,
            curriculum_fingerprint=curriculum_fp,
            sampling_fingerprint=sampling_fp,
            packing_session=p_session,
            curriculum_session=c_session,
            packing_context=p_ctx,
            curriculum_context=c_ctx,
        )

    def _build_packing_statistics(
        self,
        batches: tuple[TokenBatch, ...],
        *,
        packing_policy: PackingPolicy,
        examples_input: int,
        backend_key: str,
        overflow_deferred: int,
        overflow_truncated: int,
    ) -> PackingStatistics:
        tokens_content = 0
        tokens_padded = 0
        sequences = 0
        examples_packed = 0
        for batch in batches:
            for mask in batch.attention_mask:
                tokens_content += sum(1 for m in mask if m == 1)
                tokens_padded += sum(1 for m in mask if m == 0)
            sequences += len(batch.example_ids)
            for spans in batch.metadata.get("packed_spans", ()) if batch.metadata else ():
                examples_packed += len(spans)
        if examples_packed == 0:
            examples_packed = examples_input
        total = tokens_content + tokens_padded
        pad_ratio = (tokens_padded / total) if total else 0.0
        mean_ex = (examples_packed / sequences) if sequences else 0.0
        fingerprint = _sha(
            backend_key,
            str(packing_policy.max_sequence_length),
            _canonical(
                {
                    "content": tokens_content,
                    "padded": tokens_padded,
                    "sequences": sequences,
                    "examples": examples_packed,
                    "batches": [
                        {
                            "ids": list(b.example_ids),
                            "input_ids": [list(r) for r in b.input_ids],
                        }
                        for b in batches
                    ],
                }
            ),
        )
        return PackingStatistics(
            packing_fingerprint=fingerprint,
            backend_key=backend_key,
            examples_input=examples_input,
            examples_packed=examples_packed,
            sequences_emitted=sequences,
            tokens_content=tokens_content,
            tokens_padded=tokens_padded,
            pad_ratio=pad_ratio,
            mean_examples_per_sequence=mean_ex,
            max_sequence_length=packing_policy.max_sequence_length,
            overflow_deferred=overflow_deferred,
            overflow_truncated=overflow_truncated,
        )
