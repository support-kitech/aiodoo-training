"""Phase 6 tracking builders."""

from __future__ import annotations

from pathlib import Path

from aiodoo_training.domain.cli_profile import CLIProfile
from aiodoo_training.domain.enums import TrackingHealthStatus
from aiodoo_training.domain.experiment_session import ExperimentSession
from aiodoo_training.domain.identifiers import ExperimentId
from aiodoo_training.domain.run_record import RunRecord
from aiodoo_training.domain.tracking_policies import (
    LoggingPolicy,
    ReportPolicy,
    RetentionPolicy,
    TrackingCapability,
    TrackingHealth,
    TrackingPolicy,
)
from aiodoo_training.exceptions import BuilderError
from aiodoo_training.factories.factories import TrackerFactory
from aiodoo_training.infrastructure.tracking import capability_for
from aiodoo_training.tracking.core import (
    TrackingContext,
    TrackingStore,
    new_experiment_session,
    new_run_record,
)
from aiodoo_training.tracking.provenance import build_provenance


class TrackingBuilder:
    """Assembles TrackingPolicy / TrackingContext pieces."""

    def __init__(self) -> None:
        self._policy = TrackingPolicy()
        self._capability: TrackingCapability | None = None
        self._root: Path | None = None
        self._name = "experiment"
        self._experiment_id: ExperimentId | None = None
        self._config_fp = ""
        self._model_fp = ""
        self._adapter_fp = ""
        self._execution_digest = ""

    def with_policy(self, policy: TrackingPolicy) -> TrackingBuilder:
        self._policy = policy
        return self

    def with_root(self, root: Path) -> TrackingBuilder:
        self._root = root
        return self

    def with_identity(
        self,
        *,
        experiment_id: ExperimentId,
        name: str,
        config_fingerprint: str = "",
        model_fingerprint: str = "",
        adapter_fingerprint: str = "",
        execution_digest: str = "",
    ) -> TrackingBuilder:
        self._experiment_id = experiment_id
        self._name = name
        self._config_fp = config_fingerprint
        self._model_fp = model_fingerprint
        self._adapter_fp = adapter_fingerprint
        self._execution_digest = execution_digest
        return self

    def build_policy(self) -> TrackingPolicy:
        return self._policy

    def build_context(
        self,
        *,
        experiment_session: ExperimentSession | None = None,
        run_record: RunRecord | None = None,
        logging_policy: LoggingPolicy | None = None,
        report_policy: ReportPolicy | None = None,
        retention_policy: RetentionPolicy | None = None,
        cli_profile: CLIProfile | None = None,
    ) -> TrackingContext:
        if self._experiment_id is None and experiment_session is None:
            raise BuilderError("TrackingBuilder requires experiment identity.")
        exp = experiment_session or new_experiment_session(
            experiment_id=self._experiment_id,  # type: ignore[arg-type]
            name=self._policy.experiment_name or self._name,
            config_fingerprint=self._config_fp,
            model_fingerprint=self._model_fp,
            adapter_fingerprint=self._adapter_fp,
        )
        run = run_record or new_run_record(experiment_id=exp.experiment_id)
        root = self._root or self._policy.root_dir or Path("artifacts/tracking")
        cap = self._capability or capability_for(self._policy.backend_key)
        provenance = build_provenance(
            config_fingerprint=self._config_fp or exp.config_fingerprint,
            execution_digest=self._execution_digest,
            model_fingerprint=self._model_fp or exp.model_fingerprint,
            adapter_fingerprint=self._adapter_fp or exp.adapter_fingerprint,
        )
        run = run.with_fingerprints(provenance_digest=provenance.digest)
        return TrackingContext(
            policy=self._policy,
            capability=cap,
            health=TrackingHealth(
                backend_key=self._policy.backend_key,
                status=TrackingHealthStatus.HEALTHY,
            ),
            experiment_session=exp,
            run_record=run,
            root_dir=root,
            provenance=provenance,
            logging_policy=logging_policy or LoggingPolicy(),
            report_policy=report_policy or ReportPolicy(),
            retention_policy=retention_policy or RetentionPolicy(),
            cli_profile=cli_profile or CLIProfile(),
        )

    def build_store(self, root: Path | None = None) -> TrackingStore:
        return TrackingStore(root or self._root or Path("artifacts/tracking"))

    def create_tracker(self) -> object:
        return TrackerFactory().create(self._policy.backend_key)
