# Contract Adoption (Phase 3)

Scope: how `aiodoo-training` consumes `aiodoo_contract` (the canonical
Capability Contract package — see `ecosystem-v2-certification/
ARCHITECTURE_FREEZE_REPORT.md` and the ADRs in `aiodoo-contract/docs/adr/`).

`aiodoo-training` is the **second** canonical consumer of `aiodoo_contract`,
after `aiodoo-datasets` (see `aiodoo-datasets/CONTRACT_ADOPTION.md`). This
document records what was adopted, how, and why, and — per the
ADR-0005/ADR-0007 ownership rules ("everything shared lives in exactly one
place") — every case where this repository still defines something of its
own, with the reason it is not a duplicate contract.

Phases 0–7 (see `docs/frozen_public_contracts.md`) are permanently frozen.
Everything below is a **surgical, additive** integration on top of those
frozen ports/interfaces — no frozen port signature changed.

## 1. What "adoption" means here

`aiodoo-training` does not import `aiodoo-datasets` as a Python dependency
(there is no training → datasets dependency edge — see
`docs/adr/0008-dependency-graph.md` in `aiodoo-contract`). It consumes
datasets purely as on-disk protocol JSONL, read by
`aiodoo_training.datasets.reader.ProtocolRecordReader`. The record *shape*
written by `aiodoo-datasets`' generators is the data contract between the
two repositories, so training re-implements the same narrow projection
step aiodoo-datasets performs on the producer side — against the one
canonical target (`aiodoo_contract.schemas`), never redefining it.

### `aiodoo_training/contract/` — the adapter/bridge package

New package, training's consumer-side equivalent of aiodoo-datasets'
`generators/common/contract/`:

- **`adapters.py`** — `project_<capability>(record) -> ContractProjection`
  for every capability with a canonical shape (`planner`, `coding`,
  `repair`, `execution`, `conversation`, `approval`, `evaluation`), mapping
  a raw JSONL record onto the exact `aiodoo_contract.schemas.<capability>`
  Pydantic request/response models. Field-mapping decisions are kept
  identical to aiodoo-datasets' own adapter so a record projects the same
  way regardless of which repository reads it. Malformed input raises
  `ContractAdapterError` (never a bare `KeyError`/`TypeError`).
- **`prompt_bridge.py`** — the *only* place in this repository that turns a
  capability request into prompt text. `render_capability_prompt()`
  delegates every formatting decision to
  `aiodoo_contract.prompts.CapabilityPromptBuilder`. `serialize_response()`
  serializes the projected `CapabilityResponse` to the canonical,
  deterministic JSON text a model is trained to produce — the exact shape
  `aiodoo_contract.parsers` will later decode at runtime, rather than the
  richer, training-pedagogy-shaped `record["output"]` blob it was derived
  from. Teaching the raw dataset blob instead of the contract response is
  precisely the "training teaches one schema, runtime validates another"
  defect class this contract exists to close.
- **`version_check.py`** — `TRAINING_CONTRACT_VERSION` (the pinned contract
  version this integration was built and tested against) and
  `ensure_contract_compatible()`, which wraps
  `aiodoo_contract.version.check_compatibility`. Called at the top of
  `run_train_from_config()` (Section 4) — never invented locally.

## 2. Prompt Builder

**Removed:** every capability formatter's ad-hoc instruction/context string
assembly (`datasets/formatters/formatters.py`'s `PlannerFormatter`,
`CodingFormatter`, `RepairFormatter`, `ExecutionFormatter`,
`ConversationFormatter`, `ApprovalFormatter`, `EvaluationFormatter` each
previously owned custom prompt text).

**Replaced with:** `_ContractFormatter._format()` — shared by all seven —
which projects the record via `aiodoo_training.contract.adapters` and
renders it via `aiodoo_training.contract.prompt_bridge.build_training_example()`,
which calls `aiodoo_contract.prompts.CapabilityPromptBuilder` exclusively.
Training no longer owns a prompt builder of any kind.

Observable effect: every contract-mapped example now carries a leading
`system` turn (the capability's default system prompt from
`CapabilityPromptBuilder`) that did not exist before — training and
runtime inference now render capability prompts identically, which is the
point. Golden tokenization digests and formatter tests were regenerated /
updated to reflect this (see Section 6).

## 3. Chat Template

**Removed:** local, hand-written role-tag formatting.
`infrastructure/huggingface/templates.py` previously implemented
`SimpleRoleChatTemplate` (`<|role|>\ncontent` joined by blank lines) for
Qwen/Llama/Mistral.

**Replaced with:** `ContractBackedChatTemplate`, a thin adapter that
converts the frozen `ChatTemplate` port's `dict`-based messages
(`Sequence[Mapping[str, str]]`) to `aiodoo_contract.templates.messages.ChatMessage`
and delegates rendering to `aiodoo_contract.templates.get_chat_template(name).render_conversation(...)`.
`QwenChatTemplate` and the new `DeepSeekChatTemplate` delegate to the
contract's dedicated Qwen/DeepSeek renderers. `GenericChatTemplate` delegates
to the contract's generic renderer.

**The frozen `ChatTemplate` port itself is unchanged** — it defines
`render(messages) -> str`, and its concrete implementations continue to
satisfy it; only *what happens inside* `render()` changed, from
hand-rolled formatting to a contract delegate call. This is the one place
in the repository allowed to touch prompt-formatting internals without
being "training owning a prompt format": the port abstracts over *which*
family renders, `aiodoo_contract.templates` decides *how*.

**Llama / Mistral — documented, intentional gap.** `aiodoo_contract` does
not yet define dedicated Llama or Mistral chat templates (only
`qwen`/`deepseek`/`generic` — see
`aiodoo_contract/templates/registry.py`). `LlamaChatTemplate` and
`MistralChatTemplate` are retained as registry entries (their `family`
identity is still needed by `ModelProfile` resolution) but now delegate to
the contract's `generic` template rather than inventing a training-local
Llama/Mistral format. When `aiodoo_contract` adds dedicated templates for
these families, only the `contract_template_name=` argument in
`templates.py` needs to change.

## 4. Versioning

`aiodoo_training.contract.version_check.ensure_contract_compatible()` is
called at the top of `application/train_orchestrator.run_train_from_config()`
— before bootstrapping any registry — using
`aiodoo_contract.version.check_compatibility()`. An incompatible installed
contract (different major, or a strictly newer minor than the contract
provides) raises `ContractVersionError` (a `ConfigError` subclass), which
`run_train_from_config()`'s existing exception handling turns into a
failed `ExecutionResult` rather than training against schemas/prompts the
installed contract does not actually provide. This is training's fail-early
gate — Training must verify the Contract Version, per the phase objective.

## 5. Schemas & Validation

**Schemas:** the six contract-mapped capabilities' request/response shape
is exclusively `aiodoo_contract.schemas.<capability>.*` (Section 1) —
training defines no `Request`/`Response` model of its own for any of them.

**Validation:** `datasets/validation.py`'s `DatasetValidator` retains its
training-owned structural pre-flight check (required top-level JSONL keys,
manifest protocol-version match) — this operates on the raw record shape
aiodoo-datasets writes to disk, which is not itself an `aiodoo_contract`
schema, so there is nothing to import. It now additionally projects a
sample of records (same `sample_limit` as the existing structural check)
through `aiodoo_contract.adapters.project_record` for every capability with
a canonical shape, and runs the projected request/response through
`aiodoo_contract.validators.ContractValidator` — the same schema +
capability + version validation the contract mandates for every consumer,
never a hand-rolled equivalent. A dataset that fails this now raises
`DomainError` during validation, before training ever begins, instead of
surfacing later as a `ContractAdapterError` mid-run.

## 6. Publishing — Capability Package compatibility metadata

`artifacts/publish_contract.py`'s `build_adapter_artifact_json()` /
`build_merged_artifact_json()` / `build_base_model_artifact_json()` now
also emit:

- **`contract_version`** — always present; `aiodoo_contract.version.CONTRACT_VERSION`,
  the contract version this training run's prompts/templates/schemas were
  built against. Lets a runtime refuse to load a package trained against
  an incompatible contract instead of silently misinterpreting it.
- **`capability_package_metadata`** (adapter/merged only) — the full
  canonical `aiodoo_contract.schemas.CapabilityPackageMetadata.model_dump()`,
  attached alongside (not replacing) the frozen top-level `capability_id`
  / `adapter_type` / `peft_type` / `model_family` / `architecture` fields
  aiodoo-validation/aiodoo-model already parse. Present whenever
  `capability_id` is a real `aiodoo_contract.schemas.enums.CapabilityName`
  and both `family`/`architecture` are resolvable; **absent, never
  fabricated**, for `capability_id="context"` (not itself a capability —
  see Section 7) or when the base model family/architecture could not be
  resolved from config.

This is purely additive: the frozen `artifact_type` / `protocol_major` /
`capability_id` / `adapter_type` protocol fields aiodoo-validation and
aiodoo-model already depend on are unchanged (see
`tests/contract/test_ecosystem_capability_packages.py`, whose committed
goldens under `tests/fixtures/capability_packages/protocol/v1/` were
regenerated via `tests/fixtures/capability_packages/regenerate.py` to
include the two new fields).

Publishing remains fail-closed — see Section 8 (ACT-101).

## 7. Duplication that was **not** removed, and why

Per the primary goal ("if duplication cannot be removed, document the
reason"), the following were audited against `aiodoo_contract` and
deliberately **not** replaced with imports:

- **`domain/enums.py::DatasetType`** vs.
  `aiodoo_contract.schemas.enums.CapabilityName` — `DatasetType` is a
  pervasive, frozen (Phase 0/1) training domain type threaded through
  `DatasetRef`, dataset fingerprinting, the formatter registry, tokenization
  caching, and CLI config — replacing it with `CapabilityName` would be an
  architectural redesign of a frozen abstraction, which this phase is
  explicitly forbidden from doing. It is also not a strict duplicate:
  `DatasetType` includes `context` and `mixed`, which are not capability
  request/response shapes. The seven values that overlap with
  `CapabilityName` (`planner`/`coding`/`repair`/`execution`/`conversation`/
  `approval`/`evaluation`) are kept in lockstep by convention and enforced
  by `aiodoo_training.contract.adapters._PROJECTORS`/`SUPPORTED_CAPABILITIES`
  using the identical string values — see `_ContractFormatter`'s docstring
  in `formatters.py`.
- **`evaluation` is a first-class contract capability** —
  Training consumes certified Evaluation v2 judgment SFT
  (`evaluation_dataset.jsonl`: `candidate` / `expectation` / `rubric` →
  `verdict` / `score` / `explanation`) via `project_evaluation` and
  `EvaluationFormatter` (`_ContractFormatter`). The separate
  `evaluation_benchmark_catalog.jsonl` artifact remains
  certification/regression-only (`metadata.training_forbidden=true`) and
  is rejected by `DatasetValidator` — it is not LoRA training data.
- **`context` dataset type is not a capability** — `context` (retrieval
  results for a query) is infrastructure that other capabilities consume,
  not itself something `aiodoo_contract.schemas` defines a request/response
  for. `ContextFormatter` keeps its prior formatting; `capability_id="context"`
  is one of `TRAINING_IDS` (Section 6 fallback resolution target) but is
  never a valid `aiodoo_contract.schemas.enums.CapabilityName`, so its
  Capability Package `artifact.json` never carries the
  `capability_package_metadata` block (Section 6) — by design, not by
  omission.
- **Parsers** — `aiodoo_contract.parsers` (`CapabilityParser` and friends)
  parse a capability response's *rendered text* back into a typed model —
  the runtime-inference direction. Training never runs inference or
  consumes a model's generated output at training time; it only produces
  training labels, which is the *inverse* direction
  (`prompt_bridge.serialize_response()`, Section 1). There is no
  parsing-duplicate to remove because training performs no capability
  response parsing anywhere in its pipeline.
- **`training/checkpoint_manager.py::SaveCheckpointRequest`,
  `domain/{model_info,adapter_info,run_record,checkpoint_manifest}.py`'s
  `*Metadata` dataclasses** — these describe training's own internal
  domain objects (checkpoint save inputs, model/adapter catalog entries,
  run tracking, checkpoint manifests). None of them are a capability
  request/response or a Capability Package metadata shape;
  `aiodoo_contract` has no equivalent for any of them.

## 8. Reliability fixes (training-specific items from the architecture audits)

| ID | Finding | Fix | Test |
| :--- | :--- | :--- | :--- |
| ACT-101 | `pipeline/artifact_hooks.py::maybe_publish_artifacts` logged every publish failure — including a failed **adapter** publish, the run's core deliverable — as a warning and always let the pipeline report success. A completed training run with a broken/misconfigured publish step could report `TrainingStatus.SUCCEEDED` with no usable output. | `maybe_publish_artifacts` now returns `bool` (`False` when publishing is configured, training completed, and the adapter could not be published — no checkpoint, or the publish call raised). `_experiment_success()` folds this into `summary.json`'s `success` field, and `pipeline/handlers.py::FinalizeStage` now returns a `StageResult(status=StageStatus.FAILED)` when publish reports `False`, which `Pipeline.run()` already turns into an overall `TrainingStatus.FAILED`. Base-model / merged-model / config-snapshot publish failures remain best-effort (derived/optional artifacts). | `tests/unit/test_artifact_hooks.py` (`test_publish_error_is_logged_and_fails_closed`, `test_fails_closed_when_no_checkpoint_to_publish`, `test_finalize_stage_fails_closed_when_adapter_publish_fails`, plus the "not required" / success-path counterparts) |
| ACT-118 | `tracking/core.py::TrackingContext._degrade` and every hook in `pipeline/tracking_hooks.py` (`maybe_open_tracking`, `maybe_observe_progress`, `maybe_observe_evaluation`, `maybe_observe_export`, `maybe_finalize_tracking`) caught tracking-sink exceptions with a bare `except Exception:` and discarded them — a broken tracker (e.g. bad credentials, network partition) degraded silently with zero operator-visible signal. | `_degrade` now logs `logger.warning(...)` with the backend/run id and consecutive-failure count. Every hook's `except Exception:` now logs `logger.warning(..., exc_info=True)` before continuing (still non-fatal — tracking is intentionally best-effort — but never silent). | Existing tracking test suites (`tests/unit/test_tracking_*.py`, `tests/unit/test_pipeline_tracking_hooks.py`) continue to pass with the added logging; no behavior change to tracking degrade semantics itself, only observability. |

## 9. Deferred / out of scope for this phase

- **ACT-110 (silent default trainer backend)** — `config/training_config.py`
  defaults `backend: str = "stub"`. A misconfigured production run that
  omits `trainer.backend` silently trains against the deterministic stub
  instead of a real backend. Fixing this correctly needs a "profile"
  concept (dev/test vs. production) that does not exist in `ConfigSystem`
  today; introducing one is an architectural addition outside this phase's
  scope ("do not redesign architecture"). Tracked for a future phase with a
  dedicated, reviewed design for how profiles interact with the frozen
  config surface.
- **ACT-111 (merged Capability Package self-describing dependencies)** —
  merged packages intentionally omit `base_artifact_id`/`adapter_artifact_ids`
  (the caller/`aiodoo-model` supplies them — see
  `test_merged_not_self_describing_for_model_deps`). Making merged packages
  self-describing would require changing the frozen `ExportManifest`
  structure, which has cross-repository impact on `aiodoo-model` and is out
  of scope for a training-only phase.
- Broadening `DatasetValidator`'s new contract-validation pass (Section 5)
  beyond its existing `sample_limit` (default 32 records) to validate an
  entire dataset file on every load — left at sample-based validation to
  match the existing structural check's cost profile; not a correctness
  gap (the formatter layer still validates every record via the same
  Pydantic models at format time, just later in the pipeline).
- Any change to `aiodoo-contract`, `aiodoo-datasets`, `aiodoo-validation`,
  `aiodoo-model`, `aiodoo-core`, `aiodoo-vscode`, `aiodoo-colab` — out of
  scope per this phase's instructions.

## 10. Backward compatibility

- The frozen `ChatTemplate`, `ExampleFormatter`, `DatasetSource` ports, and
  the `chat_template_registry` / `formatter_registry` registration
  mechanism, are unchanged. Every existing call site (`tokenization/pipeline.py`,
  `datasets/source.py`) works unmodified.
- **Intentional, observable change:** the six contract-mapped dataset
  types' `TrainingExample.messages` now include a leading `system` turn
  (Section 2), so downstream consumers asserting an exact `(user,
  assistant)` message count for `coding`/`planner`/`repair`/`execution`/
  `conversation`/`approval` need to expect `(system, user, assistant)`
  instead. `context`/`evaluation` are unchanged (still `(user, assistant)`).
  Golden tokenization digests were regenerated to reflect the new prompt
  text (`tests/golden/coding_tokens.sha256`); this is the audit-mandated
  "training and runtime must produce identical prompt formatting" fix, not
  a regression.
- **Intentional, observable change (ACT-101):** a training run that
  previously reported `success: true` in `summary.json` / a `SUCCEEDED`
  `FinalizeStage` result despite a failed adapter publish will now
  correctly report `success: false` / `StageStatus.FAILED`. Any external
  tooling that polled `summary.json`'s `success` field and assumed it only
  reflected training/evaluation/quality-gate outcome (not publish outcome)
  should be re-checked against this change.
- `artifact.json`'s new `contract_version` / `capability_package_metadata`
  fields are additive; no existing field was renamed or removed (see
  Section 6).
