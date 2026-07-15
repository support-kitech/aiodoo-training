"""Enumerations for the aiodoo-training domain."""

from enum import StrEnum


class TrainingStatus(StrEnum):
    """Lifecycle status of a training run."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Precision(StrEnum):
    """Numeric precision policy for model loading and training."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class DeviceKind(StrEnum):
    """Logical execution devices (framework-independent)."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    XPU = "xpu"


class AcceleratorKind(StrEnum):
    """Distributed / accelerator orchestration preference."""

    NONE = "none"
    DDP = "ddp"
    FSDP = "fsdp"
    DEEPSPEED = "deepspeed"
    ACCELERATE = "accelerate"
    XLA = "xla"


class DistributedStatus(StrEnum):
    """Lifecycle status of a DistributedSession (not TrainingStatus)."""

    PENDING = "pending"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    DRAINING = "draining"
    FAILED = "failed"
    COMPLETED = "completed"
    ABORTED = "aborted"


class WorkerStatus(StrEnum):
    """Per-rank runtime observation (never TrackingHealth)."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    STALLED = "stalled"
    LOST = "lost"
    DRAINING = "draining"


class NodeStatus(StrEnum):
    """Per-node runtime observation."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    LOST = "lost"


class ClusterStatus(StrEnum):
    """Whole-mesh runtime observation."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    ORPHANED = "orphaned"


class DistributedCheckpointMode(StrEnum):
    """Who writes / how shards are organized (not Phase 3 CheckpointPolicy)."""

    RANK0_FULL = "rank0_full"
    SHARDED = "sharded"
    HYBRID = "hybrid"


class RankRole(StrEnum):
    """Role assigned by DistributedCheckpointCoordinator for a save."""

    COORDINATOR = "coordinator"
    SHARD_WRITER = "shard_writer"
    REPLICA = "replica"
    IDLE = "idle"


class ReductionOp(StrEnum):
    """Cross-rank reduction operator (framework-independent)."""

    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


class BarrierTimeoutAction(StrEnum):
    """Policy when a barrier times out."""

    FAIL = "fail"
    WARN_CONTINUE = "warn_continue"


class RestartFrom(StrEnum):
    """Where RestartPolicy relaunches from (ResumePolicy still gates ckpt)."""

    LAST_CKPT = "last_ckpt"
    SCRATCH = "scratch"


class AdapterType(StrEnum):
    """Parameter-efficient or full adaptation strategy."""

    LORA = "lora"
    QLORA = "qlora"
    FULL = "full"
    NONE = "none"


class ExportType(StrEnum):
    """
    Supported export artifact kinds.

    ``PEFT_ADAPTER`` names an artifact format (adapter weights + config), not a
    dependency on the PEFT library.
    """

    PEFT_ADAPTER = "peft_adapter"
    MERGED_WEIGHTS = "merged_weights"
    TOKENIZER = "tokenizer"
    MANIFEST = "manifest"
    BUNDLE = "bundle"


class DatasetType(StrEnum):
    """Generator / dataset family consumed from aiodoo-datasets."""

    PLANNER = "planner"
    CODING = "coding"
    REPAIR = "repair"
    CONTEXT = "context"
    EXECUTION = "execution"
    APPROVAL = "approval"
    CONVERSATION = "conversation"
    EVALUATION = "evaluation"
    MIXED = "mixed"


class ModelFamily(StrEnum):
    """Supported base model families (extensible via registry + templates)."""

    QWEN = "qwen"
    LLAMA = "llama"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    GEMMA = "gemma"
    PHI = "phi"
    UNKNOWN = "unknown"


class TrainingBackend(StrEnum):
    """Concrete trainer backend identifiers."""

    HF_TRAINER = "hf_trainer"
    CUSTOM_LOOP = "custom_loop"


class CheckpointType(StrEnum):
    """Kinds of persisted training state."""

    FULL_STATE = "full_state"
    ADAPTER_ONLY = "adapter_only"
    METRICS_ONLY = "metrics_only"


class TrackerType(StrEnum):
    """Experiment tracking sink identifiers."""

    NULL = "null"
    LOCAL_JSONL = "local_jsonl"
    MLFLOW = "mlflow"
    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
    OTEL = "otel"


class ExperimentStatus(StrEnum):
    """Lifecycle status of an experiment catalog entry."""

    PENDING = "pending"
    ACTIVE = "active"
    ARCHIVED = "archived"
    FAILED = "failed"


class RunState(StrEnum):
    """Observational mirror of a pipeline run outcome (not TrainingStatus)."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    RESUMED = "resumed"


class TrackingSinkStatus(StrEnum):
    """Open-run lifecycle status of a tracking sink session."""

    CLOSED = "closed"
    OPEN = "open"
    FLUSHING = "flushing"
    DEGRADED = "degraded"


class TrackingHealthStatus(StrEnum):
    """Backend sink health for diagnostics — never training health."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    READ_ONLY = "read_only"
    OFFLINE = "offline"
    FAILED = "failed"


class CLIProfileName(StrEnum):
    """Named CLI UX presets."""

    DEFAULT = "default"
    MINIMAL = "minimal"
    VERBOSE = "verbose"
    JSON = "json"
    CI = "ci"


class ArtifactRelationKind(StrEnum):
    """Typed edge labels for artifact lineage."""

    PRODUCED_BY = "produced_by"
    EVALUATES = "evaluates"
    EXPORTS = "exports"
    RESUMES = "resumes"


class PipelineStage(StrEnum):
    """Named stages of the training pipeline."""

    VALIDATE_CONFIG = "validate_config"
    BOOTSTRAP_DETERMINISM = "bootstrap_determinism"
    RESOLVE_EXECUTION = "resolve_execution"
    ASSEMBLE_DATASETS = "assemble_datasets"
    TOKENIZE = "tokenize"
    LOAD_MODEL = "load_model"
    APPLY_ADAPTATION = "apply_adaptation"
    PLAN_PACKING = "plan_packing"
    PLAN_CURRICULUM = "plan_curriculum"
    CREATE_TRAINER = "create_trainer"
    RESTORE_CHECKPOINT = "restore_checkpoint"
    TRAIN = "train"
    EVALUATE = "evaluate"
    EXPORT = "export"
    FINALIZE = "finalize"


class StageStatus(StrEnum):
    """Outcome of an individual pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class PackingMode(StrEnum):
    """Sequence packing strategies."""

    NONE = "none"
    CONCAT = "concat"
    BEST_FIT = "best_fit"
    LENGTH_AWARE = "length_aware"


class CurriculumMode(StrEnum):
    """Curriculum scheduling strategies."""

    SEQUENTIAL = "sequential"
    WEIGHTED_MIX = "weighted_mix"
    NONE = "none"
    DIFFICULTY = "difficulty"
    RANDOM = "random"
    MIXED = "mixed"


class PackingStatus(StrEnum):
    """Lifecycle status of a packing plan build."""

    PENDING = "pending"
    PLANNING = "planning"
    READY = "ready"
    FAILED = "failed"
    SKIPPED = "skipped"


class CurriculumStatus(StrEnum):
    """Lifecycle status of a curriculum plan."""

    PENDING = "pending"
    PLANNING = "planning"
    READY = "ready"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PackingOverflow(StrEnum):
    """Overflow policy when an example does not fit the current sequence."""

    DEFER = "defer"
    TRUNCATE = "truncate"
    REJECT = "reject"


class EvaluationStatus(StrEnum):
    """Lifecycle status of an evaluation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class ExportStatus(StrEnum):
    """Lifecycle status of an export attempt."""

    PENDING = "pending"
    VALIDATING = "validating"
    PACKAGING = "packaging"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetSplitKind(StrEnum):
    """Held-out dataset split semantics for evaluation."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    BENCHMARK = "benchmark"
    CUSTOM = "custom"


class ComparisonOp(StrEnum):
    """Comparison operators for quality thresholds."""

    LE = "<="
    GE = ">="
    EQ = "=="
    LT = "<"
    GT = ">"
