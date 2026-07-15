"""Phase 7 distributed builders."""

from __future__ import annotations

from aiodoo_training.distributed.context import DistributedContext
from aiodoo_training.distributed.runtime import DistributedRuntime
from aiodoo_training.domain.config import DistributedSpec
from aiodoo_training.domain.distributed_policies import DistributedRuntimePolicy
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.exceptions import BuilderError


class DistributedContextBuilder:
    """Assembles a DistributedContext via DistributedRuntime.open."""

    def __init__(self) -> None:
        self._policy = DistributedRuntimePolicy()
        self._spec = DistributedSpec()
        self._execution: ExecutionEnvironment | None = None
        self._runtime = DistributedRuntime()

    def with_policy(self, policy: DistributedRuntimePolicy) -> DistributedContextBuilder:
        self._policy = policy
        return self

    def with_spec(self, spec: DistributedSpec) -> DistributedContextBuilder:
        self._spec = spec
        return self

    def with_execution(self, execution: ExecutionEnvironment) -> DistributedContextBuilder:
        self._execution = execution
        return self

    def build(self) -> DistributedContext:
        if self._execution is None:
            raise BuilderError("DistributedContextBuilder requires ExecutionEnvironment.")
        return self._runtime.open(self._policy, self._execution, self._spec)

    @property
    def runtime(self) -> DistributedRuntime:
        return self._runtime
