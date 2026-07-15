"""Phase 5 curriculum builders."""

from __future__ import annotations

from typing import Any

from aiodoo_training.curriculum import CurriculumContext
from aiodoo_training.domain.config import CurriculumSpec
from aiodoo_training.domain.curriculum_session import CurriculumSession
from aiodoo_training.domain.enums import CurriculumMode
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.exceptions import BuilderError


class CurriculumPlanBuilder:
    """Assembles :class:`CurriculumSpec` from mode / stages / seed."""

    def __init__(self) -> None:
        self._mode = CurriculumMode.NONE
        self._stages: list[str] = []
        self._backend_key = "none"
        self._seed = 42

    def with_mode(self, mode: CurriculumMode | str) -> CurriculumPlanBuilder:
        self._mode = CurriculumMode(mode) if not isinstance(mode, CurriculumMode) else mode
        return self

    def with_backend(self, key: str) -> CurriculumPlanBuilder:
        self._backend_key = key
        return self

    def with_stages(self, *stages: str) -> CurriculumPlanBuilder:
        self._stages = list(stages)
        return self

    def add_stage(self, name: str) -> CurriculumPlanBuilder:
        self._stages.append(name)
        return self

    def with_seed(self, seed: int) -> CurriculumPlanBuilder:
        self._seed = seed
        return self

    def build_spec(self) -> CurriculumSpec:
        return CurriculumSpec(mode=self._mode, stages=tuple(self._stages))

    @property
    def backend_key(self) -> str:
        return self._backend_key

    @property
    def seed(self) -> int:
        return self._seed


class CurriculumContextBuilder:
    """Assembles a resolved :class:`CurriculumContext`."""

    def __init__(self) -> None:
        self._pieces: dict[str, object] = {}

    def with_piece(self, key: str, value: object) -> CurriculumContextBuilder:
        self._pieces[key] = value
        return self

    def with_session(self, session: CurriculumSession) -> CurriculumContextBuilder:
        self._pieces["curriculum_session"] = session
        return self

    def with_spec(self, spec: CurriculumSpec) -> CurriculumContextBuilder:
        self._pieces["curriculum_spec"] = spec
        return self

    def build(self) -> CurriculumContext:
        session = self._pieces.get("curriculum_session")
        spec = self._pieces.get("curriculum_spec")
        if not isinstance(session, CurriculumSession):
            raise BuilderError("CurriculumContextBuilder requires CurriculumSession.")
        if not isinstance(spec, CurriculumSpec):
            raise BuilderError("CurriculumContextBuilder requires CurriculumSpec.")
        seed_raw = self._pieces.get("seed", 42)
        seed = seed_raw if isinstance(seed_raw, int) else 42
        backend_raw = self._pieces.get("backend_key", "none")
        backend = backend_raw if isinstance(backend_raw, str) else "none"
        extra = self._pieces.get("bind_extra")
        bind_extra: dict[str, Any] = dict(extra) if isinstance(extra, dict) else {}
        return CurriculumContext(
            curriculum_session=session,
            curriculum_spec=spec,
            seed=seed,
            backend_key=backend,
            bind_extra=bind_extra,
        )


def build_curriculum_session(
    *,
    experiment_id: ExperimentId,
    run_id: RunId,
    session_id: str = "cur-session",
) -> CurriculumSession:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return CurriculumSession(
        session_id=session_id,
        experiment_id=experiment_id,
        run_id=run_id,
        created_at=now,
        updated_at=now,
    )
