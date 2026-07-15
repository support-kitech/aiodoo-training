"""Shared exception hierarchy for aiodoo-training."""


class AiodooTrainingError(Exception):
    """Base error for all aiodoo-training failures."""


class DomainError(AiodooTrainingError):
    """Raised when domain invariants are violated."""


class ConfigError(AiodooTrainingError):
    """Raised when configuration loading, composition, or validation fails."""


class RegistryError(AiodooTrainingError):
    """Raised when registry registration or lookup fails."""


class PipelineError(AiodooTrainingError):
    """Raised when pipeline construction or stage execution fails."""


class DeterminismError(AiodooTrainingError):
    """Raised when deterministic fingerprint or seed setup fails."""


class BuilderError(AiodooTrainingError):
    """Raised when a builder cannot produce a valid domain object."""


class FactoryError(AiodooTrainingError):
    """Raised when a factory cannot resolve or construct a collaborator."""


class TrainingLifecycleError(AiodooTrainingError):
    """Raised when an illegal training session lifecycle transition is attempted."""


class EvaluationLifecycleError(AiodooTrainingError):
    """Raised when an illegal evaluation session lifecycle transition is attempted."""


class ExportLifecycleError(AiodooTrainingError):
    """Raised when an illegal export session lifecycle transition is attempted."""


class ExportError(AiodooTrainingError):
    """Raised when export packaging or validation fails."""


class IncompatibleResume(AiodooTrainingError):
    """Raised when a checkpoint cannot be resumed under the active policy."""


class CheckpointCorruption(AiodooTrainingError):
    """Raised when checkpoint artifacts are missing, unreadable, or tampered."""


class ArtifactValidationError(AiodooTrainingError):
    """Raised when artifact bundle integrity validation fails."""


class ArtifactIncompatible(AiodooTrainingError):
    """Raised when an artifact bundle fails consumer compatibility negotiation."""


class ArtifactCorruption(AiodooTrainingError):
    """Raised when artifact bundle files are missing, unreadable, or tampered."""


class PackingLifecycleError(AiodooTrainingError):
    """Raised when an illegal packing session lifecycle transition is attempted."""


class CurriculumLifecycleError(AiodooTrainingError):
    """Raised when an illegal curriculum session lifecycle transition is attempted."""


class PackingError(AiodooTrainingError):
    """Raised when packing planning fails (e.g. overflow reject)."""


class TrackingLifecycleError(AiodooTrainingError):
    """Raised when an illegal tracking / experiment / run lifecycle transition occurs."""


class TrackingError(AiodooTrainingError):
    """Raised when a tracking sink operation fails fatally."""


class DistributedLifecycleError(AiodooTrainingError):
    """Raised when an illegal distributed session lifecycle transition occurs."""


class DistributedError(AiodooTrainingError):
    """Raised when distributed runtime / collective operations fail."""


class ResumeWarning(Warning):
    """Non-fatal resume compatibility issue recorded by CheckpointManager."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
