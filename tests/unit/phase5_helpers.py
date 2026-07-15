"""Shared fixtures / helpers for Phase 5 packing & curriculum tests."""

from __future__ import annotations

from types import MappingProxyType

from aiodoo_training.domain.config import CurriculumSpec, PackingSpec
from aiodoo_training.domain.enums import CurriculumMode, DatasetType, PackingMode
from aiodoo_training.domain.examples import TrainingExample, freeze_messages
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.packing_policies import PackingPolicy, SamplingSpec
from aiodoo_training.factories import (
    CurriculumStrategyFactory,
    PackingStrategyFactory,
    SamplingStrategyFactory,
)
from aiodoo_training.packing.planner import SchedulePlan, SchedulePlanner


def make_examples(n: int = 5) -> tuple[TrainingExample, ...]:
    out: list[TrainingExample] = []
    for i in range(n):
        stage = "easy" if i < n // 2 else "hard"
        out.append(
            TrainingExample(
                example_id=f"e{i}",
                dataset_type=DatasetType.CODING if i % 2 == 0 else DatasetType.REPAIR,
                messages=freeze_messages(
                    [
                        {"role": "user", "content": "u" * (i + 1)},
                        {"role": "assistant", "content": "a" * (i + 1)},
                    ]
                ),
                metadata=MappingProxyType(
                    {
                        "difficulty": float(i),
                        "stage": stage,
                        "weight": 1.0 + float(i),
                    }
                ),
            )
        )
    return tuple(out)


def plan_once(
    examples: tuple[TrainingExample, ...] | None = None,
    *,
    packing_backend: str = "concat",
    curriculum_backend: str = "sequential",
    sampling_backend: str = "identity",
    packing_mode: PackingMode = PackingMode.CONCAT,
    curriculum_mode: CurriculumMode = CurriculumMode.SEQUENTIAL,
    stages: tuple[str, ...] = ("easy", "hard"),
    max_sequence_length: int = 64,
    seed: int = 42,
) -> SchedulePlan:
    ex = examples if examples is not None else make_examples()
    return SchedulePlanner().ensure_order(
        ex,
        curriculum=CurriculumStrategyFactory().create(curriculum_backend),
        sampling=SamplingStrategyFactory().create(sampling_backend),
        packing=PackingStrategyFactory().create(packing_backend),
        curriculum_spec=CurriculumSpec(mode=curriculum_mode, stages=stages),
        packing_spec=PackingSpec(mode=packing_mode, max_sequence_length=max_sequence_length),
        sampling_spec=SamplingSpec(backend_key=sampling_backend, seed=seed),
        packing_policy=PackingPolicy(
            backend_key=packing_backend,
            mode=packing_mode,
            max_sequence_length=max_sequence_length,
            seed=seed,
        ),
        experiment_id=ExperimentId(value="phase5"),
        run_id=RunId(value="run"),
        seed=seed,
    )
