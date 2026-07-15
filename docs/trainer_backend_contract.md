# TrainerBackend Contract

**Status:** Binding for all Phase 3+ trainer implementations (Phase 3 permanently frozen)  
**Related:** [Phase 3 Architecture](phase3-training-engine-architecture.md),
[ADR-0013](adr/0013-phase3-training-engine.md), [ADR-0014](adr/0014-phase3-freeze.md),
[Frozen Public Contracts](frozen_public_contracts.md)

Every concrete `TrainerBackend` registered in AIODOO Training **must** satisfy
this contract. Factories, boundary tests, and freeze reviews use it as the
compliance checklist. Architecture is frozen — backends implement; they do not
redesign the port.

Frozen port surface (do not widen):

```text
TrainerBackend.train(config, model, execution) -> TrainingProgress
TrainerBackend.resume(config, model, checkpoint, execution) -> TrainingProgress
```

Richer runtime state reaches the backend only via factory construction /
`bind(TrainingContext)` (or equivalent binder). Never add parameters to the
frozen ABC.

---

## 1. Lifecycle integration

- Drive session status only through application `TrainingLifecycle` (or
  equivalent COW helpers that enforce the same transition rules).
- Allowed transitions follow frozen `TrainingStatus`:
  `PENDING → RUNNING`, `RUNNING ↔ PAUSED`, `RUNNING → COMPLETED|FAILED|CANCELLED`,
  and cancel paths from `PENDING` / `PAUSED`.
- Do not mutate `TrainingSession` in place. Always copy-on-write.
- Emit `TrainingFailed` (or equivalent failure event) before re-raising on
  unrecoverable errors.
- Pause / interrupt hooks (tests, operator stop) leave the session in
  `PAUSED` with a durable checkpoint when policy requires it — not a silent
  half-updated status.

---

## 2. TrainingSession ownership

- Treat `TrainingSession` as the run cursor: identity, `global_step`, `epoch`,
  fingerprints, embedded `DatasetSession`, resume path metadata.
- Derive emitted `TrainingProgress` from the session plus collected metrics;
  do not maintain a divergent step counter that disagrees with the session.
- On resume, align session `global_step` / `epoch` with the validated
  checkpoint (via `CheckpointManager` / `ResumeCoordinator`), then continue.
- Keep session updates synchronized on the bound `TrainingContext` when one is
  present (`with_training_session` / `with_dataset_session`).

---

## 3. CheckpointManager usage

- **Never** write checkpoint filesystem packages directly from the trainer.
- Request durability only through `CheckpointManager.save(...)` (weights via
  frozen `CheckpointStore`; manifests, RNG, DatasetSession, metrics sidecars
  owned by the manager).
- Honor `CheckpointPolicy` / `CheckpointingSpec` (`save_steps`, limits,
  optional save-on-failure) without inventing parallel retention logic.
- Do not index or prune `.tmp-*` directories; incomplete publishes are manager
  concerns.
- Atomic publish, fingerprinting, and `training_protocol_version` validation
  are manager responsibilities — trainers must not bypass them.

---

## 4. ResumePolicy support

- Resume entry goes through `CheckpointManager.load_and_validate` /
  `ResumeCoordinator` with an explicit `ResumePolicy`
  (`STRICT` | `WARN` | `RELAXED`; default `STRICT`).
- Do not soft-load incompatible checkpoints when STRICT rejects them.
- After a successful validate/restore bundle:
  - restore model via `CheckpointStore.restore`
  - restore RNG via `RngController.restore`
  - restore `DatasetSession` / training cursor from sidecars / manifest
- Continue the loop with frozen `TrainerBackend.resume(config, model,
  checkpoint, execution)` — opaque handle + checkpoint path only; no new
  resume signature.

---

## 5. Deterministic behavior

Given the same configuration, dataset identity, model/adapter fingerprints,
seed, trainer backend key, and resolved `ExecutionEnvironment`, and with
`ResumePolicy.STRICT` compatibility against the checkpoint:

> A resumed run must produce the same deterministic progression as an
> uninterrupted run (global step, epoch, recorded training metrics such as
> loss for steps after the resume point).

Mandatory practices:

- Seed through `RngController` before the first step of a fresh run.
- Restore RNG snapshots before the first step after resume.
- Prefer deterministic data ordering / DatasetSession cursors over hidden
  global RNG draws for loss computation when under stub or bit-exact CI.
- Document any device-class limits (e.g. cross-GPU SKU non-bit-exact) without
  violating portable experiment fingerprints.

---

## 6. ResourcePlanner usage

- Never invent device / precision / memory / accelerator decisions inside the
  trainer.
- Consume the already-resolved `ExecutionEnvironment` passed into
  `train` / `resume` (produced by frozen `ResourcePlanner`).
- Do not bypass planner policy (e.g. forcing CUDA when environment says CPU).
- Respect opaque execution digests used in checkpoint compatibility checks.

---

## 7. Opaque handle usage

- Accept and return only AIODOO opaque handles (`TrainableModelHandle`,
  `CheckpointHandle`, and additive optimizer/scheduler handles when used).
- Do not expose Torch, Transformers, PEFT, bitsandbytes, or Accelerate objects
  on public method signatures, domain types, or application-layer DTOs.
- Framework objects remain private inside infrastructure carriers.

---

## 8. Framework isolation

- Backend implementations live under `aiodoo_training/infrastructure/...`.
- Static and dynamic imports of forbidden frameworks outside
  `infrastructure/` are a contract violation (enforced by boundary tests).
- Optional heavy deps (e.g. Transformers) must fail clearly when missing;
  CPU stub backends must remain usable for CI without train extras.

---

## 9. Event emission

Publish domain `TrainingEvent` values (via bound `TrainingEventBus` when
present) for the happy path and failure path, including at least:

- `TrainingStarted`
- step / loss progression (`StepStarted`, `LossComputed`, `StepCompleted` as
  applicable)
- `CheckpointCreated` after a successful manager publish
- `TrainingCompleted` or `TrainingFailed`

Events carry `experiment_id`, `run_id`, `session_id`, timestamps, and step /
epoch fields. Callbacks receive `CallbackContext` and must not mutate
sessions in place.

---

## 10. Boundary compliance

A backend is compliant only if **all** of the following hold:

| Check | Requirement |
|-------|-------------|
| Port surface | Uses frozen `train` / `resume` signatures only |
| Session | Uses `TrainingSession` + lifecycle COW updates |
| Checkpoints | Uses `CheckpointManager`; never bypasses store protocol |
| Resume | Supports `ResumePolicy` via manager / coordinator |
| Determinism | Preserves resume-equivalence golden invariant on CPU stub |
| Resources | Uses resolved `ExecutionEnvironment`; never bypasses planner |
| Handles | Opaque AIODOO handles only |
| Isolation | No framework types leak outside infrastructure |
| Events | Emits training events on the ordered bus |
| Tests | Passes architecture boundary tests and backend-specific unit / resume tests |

---

## Registration

Register backends with `trainer_registry` under stable keys (e.g. `stub`,
`hf_trainer`). `TrainerBackendFactory.create(key)` constructs them; Phase 3
bootstraps defaults via `bootstrap_phase3()`.

New backends (custom loop, DeepSpeed, FSDP, Accelerate) **extend by
registration**, not by changing this contract or the frozen port.
