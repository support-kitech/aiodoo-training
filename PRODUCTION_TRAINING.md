# AIODOO Production Adapter Training — Readiness Report

**Phase:** Milestone B, Phase 10 — Production Adapter Training Readiness
**Date:** 2026-07-21
**Scope:** This document certifies that the AIODOO ecosystem — `aiodoo-contract`
→ `aiodoo-datasets` → `aiodoo-training` → `aiodoo-validation` → `aiodoo-model`
→ `aiodoo-core` → `aiodoo-vscode` → `aiodoo-colab`, all frozen — is ready to
launch production adapter training for each of the 8 capability packs defined
under `configs/training/`. This is a readiness verification, not a redesign:
no architecture, ownership, contract, or algorithm was changed to produce it.
No production training, packaging, or merge was performed by this phase.

---

## 1. Training architecture

Each of the 8 rows in `configs/training/README.md`'s catalog is an
**independent capability adapter**: a fresh QLoRA trained from the same base
model (`Qwen/Qwen3-8B`), never chained on top of another product's adapter.
`checkpointing.resume_from` exists solely for **same-run** recovery after an
interruption (e.g. a Colab disconnect) — never for cross-capability weight
chaining. Product-level composition (e.g. combining `aiodoo-coding` +
`aiodoo-planner` into a "Development" product) is explicitly out of scope for
`aiodoo-training` and belongs to `aiodoo-model`'s merge/promotion pipeline
(itself out of scope for this phase, per its explicit instruction to stop
before packaging/merge).

```text
configs/training/<capability>/
├── dataset.yaml      # dataset file, dataset_version, stage/adapter naming
├── model.yaml        # base model identity (Qwen/Qwen3-8B for all 8)
├── training.yaml     # trainer/adaptation/precision/packing/checkpointing/resume
├── evaluation.yaml    # eval gate + acceptance thresholds
├── export.yaml       # artifact roles, output dir, directory-naming conventions
├── experiment.yaml   # root include (`python train.py --config .../experiment.yaml`)
└── README.md
```

Invocation is identical for all 8 capabilities:

```bash
python train.py --config configs/training/<capability>/experiment.yaml
```

`aiodoo-colab`'s `ExperimentStore` (`python/experiments.py`) discovers and
loads this exact directory layout directly from a cloned `aiodoo-training`
checkout (with a Drive-override layer on top for path overlays) — verified by
reading `_canonical_training_config`/`ExperimentStore.load`, which reads the
same `dataset.yaml`/`model.yaml`/`training.yaml`/`evaluation.yaml`/
`export.yaml` fragments this document audits. No duplicate config parser
exists in `aiodoo-colab`.

## 2. Capability matrix

| # | Capability | Dataset file | Records | Dataset scale | Config validated | Validation profile (`aiodoo-validation`) | Est. GPU (T4, 16GB) | Est. duration (T4, single session) | Adapter output |
|---:|---|---|---:|---|:---:|---|---|---|---|
| 1 | `coding` | `coding_v1_0.jsonl` | 5,459 | Training-scale | PASS | `capabilities/coding/` (full) | ~13–14 GB peak (QLoRA 4-bit, seq 2048) | ~1.5–3 h | `models/adapters/aiodoo-coding/` |
| 2 | `planner` | `planner_v1_0.jsonl` | 5,695 | Training-scale | PASS | `capabilities/planner/` (full) | ~13–14 GB | ~1.5–3 h | `models/adapters/aiodoo-planner/` |
| 3 | `execution` | `execution_dataset.jsonl` | 5,459 | Training-scale | PASS | `capabilities/execution/` (full) | ~13–14 GB | ~1.5–3 h | `models/adapters/aiodoo-execution/` |
| 4 | `repair` | `repair_v1_0.jsonl` | 481 | Training-scale (smaller) | PASS | `capabilities/repair/` (full) | ~13–14 GB | ~15–30 min | `models/adapters/aiodoo-repair/` |
| 5 | `context` | `context_v1_0.jsonl` | 50,161 | Training-scale (largest) | PASS | **N/A — infrastructure adapter, not a Capability Contract member** (see §10) | ~13–14 GB | ~10–20 h — **may exceed a single Colab session; same-run `resume_from` is required here** (see §6) | `models/adapters/aiodoo-context/` |
| 6 | `conversation` | `conversation_dataset.jsonl` | 1 | **Not training-scale — placeholder** | PASS (schema-valid, but content is a single stub record) | `capabilities/conversation/` (profile exists, corpus unusable) | N/A until dataset rebuilt | N/A until dataset rebuilt | `models/adapters/aiodoo-conversation/` |
| 7 | `approval` | `approval_dataset.jsonl` | 1 | **Not training-scale — placeholder** | PASS (schema-valid, but content is a single stub record) | `capabilities/approval/` (profile exists, corpus unusable) | N/A until dataset rebuilt | N/A until dataset rebuilt | `models/adapters/aiodoo-approval/` |
| 8 | `evaluation` | `evaluation_dataset.jsonl` | 1 | **Not training-scale — placeholder** | PASS (schema-valid, but content is a single stub record) | `capabilities/evaluation/` (profile exists, corpus unusable) | N/A until dataset rebuilt | N/A until dataset rebuilt | `models/adapters/aiodoo-evaluation/` |

All 8 configuration packs (`experiment.yaml` + fragments) pass
`python3 validate_config.py --config configs/training/<id>/experiment.yaml`
(compose → validate → resolve → fingerprint), confirmed for every row in this
table during this phase.

**Training-ready today: `coding`, `planner`, `execution`, `repair`, `context`
(5 of 8).** `conversation`, `approval`, `evaluation` have valid configuration
and a valid (schema-passing) dataset *file*, but the dataset *content* is a
single placeholder record each — this is a pre-existing, explicitly documented
limitation of `aiodoo-datasets` (its own `docs/production_freeze_report.md`:
*"Do not train on approval / conversation / evaluation at scale until
rebuilt"*), not a training-pipeline or configuration defect, and it is outside
this phase's mandate to regenerate datasets. GPU/duration estimates are
intentionally omitted for these three until real corpora exist — estimating
against a 1-record corpus would be meaningless.

Estimates above are planning-grade order-of-magnitude figures (QLoRA 4-bit,
rank 16, effective batch 16, seq 2048, on a Tesla T4), not measured
benchmarks — consistent with this ecosystem's existing documentation
practice of not claiming CI-proven numbers it hasn't actually measured (see
`aiodoo-datasets/docs/production_freeze_report.md` §5 for the same honesty
convention applied there).

## 3. Training sequence

Each of the 5 training-ready capabilities is launched **independently** —
there is no required ordering between them, since none resumes from another's
weights. A sensible operational sequence for a single-GPU Colab session,
smallest-to-largest by wall-clock cost:

```text
1. repair      (481 records    — fastest, good smoke/sanity run first)
2. coding      (5,459 records)
3. execution   (5,459 records)
4. planner     (5,695 records)
5. context     (50,161 records — longest; plan for checkpoint/resume across sessions)
```

Each run independently produces one Capability Package
(`models/adapters/aiodoo-<capability>/`); none blocks another, and any subset
can be trained in parallel across separate Colab sessions/GPUs since they
share only read access to the same base model cache and dataset files.

## 4. Hardware recommendations

All 8 `training.yaml` fragments are hand-tuned, byte-identically, for a single
**Tesla T4 (16 GB, sm_75)** — the standard free/Colab-Pro GPU tier — verified
across all 8 capabilities: `precision.compute: fp16` (T4 has no production
bf16; requires Ampere+), `adaptation.strategy: qlora` with `load_in_4bit:
true` (mandatory to fit an 8B model in ~15 GB usable VRAM), `packing.
max_sequence_length: 2048` (the documented longest stable context on T4 QLoRA
+ packing — 4096 was found to OOM), `per_device_batch_size: 1` × `gradient_
accumulation_steps: 16` (only safe micro-batch size at this scale/length),
`activation_checkpointing: true`, and `attn.preferred: sdpa` with optional
Flash-Attention 2 when the runtime image provides it.

`execution.device.allow_cpu_fallback: false` is set identically across all 8
— this is a deliberate **fail-closed** production choice: a production GPU
run must fail loudly if no GPU is attached, not silently fall back to a CPU
run that would take days and produce a result nobody asked for. Verified
against the actual `TorchResourcePlanner._select_device` implementation
(`aiodoo_training/infrastructure/resources/torch_planner.py`): when
`preferred != AUTO/CPU`, the requested device is unavailable, and
`allow_cpu_fallback` is `false`, it raises `DomainError` rather than
silently downgrading.

On a larger production GPU (A100/L4/H100-class, ≥24 GB), the same
configuration runs as-is with headroom to spare; increasing
`per_device_batch_size` and/or `max_sequence_length` on such hardware is a
future tuning opportunity, not a requirement — no config change is needed to
train correctly on better hardware than the T4 floor these configs target.

## 5. Checkpoint strategy

Identical across all 8 capabilities (verified): `save_steps: 200`,
`save_total_limit: 3` (keep the last 3 — enough to recover from a Colab
preemption without unbounded Drive growth), `save_on_failure: true`
(emergency checkpoint on a crashed stage), `validate_on_load: true`
(fingerprint/strict checks before any resume), output directory
`training/cache/<capability>/checkpoints/` under the Drive workspace. A final
checkpoint is always written at training end regardless of the `save_steps`
boundary, covering short smoke runs and trailing steps.

Checkpoint contents (verified against `configs/experiments/artifacts/
checkpoints/example-phase0/checkpoint-10/`, a committed reference fixture):
`weights.json`/`adapter_model.safetensors`, `optimizer.json`, `rng.json`,
`dataset_session.json` (data-loader position, for exact-resume), `metrics.
json`, and a `checkpoints.json` index plus a `metrics/history.jsonl` append
log — matching `CHECKPOINT_SIDECAR_FILENAMES` in `aiodoo_training/artifacts/
publish_contract.py`, which also enumerates exactly which of these are
**never** published downstream (sidecars stay in `training/cache/`; only
`adapter_model.*` + `adapter_config.json` + optional tokenizer files leave
the checkpoint directory for publish).

## 6. Resume strategy

`resume.policy: strict` is set identically across all 8: a resume with a
mismatched model/config fingerprint is refused, not silently accepted (a
`ResumeWarning`-only "warn" policy also exists and is exercised by
`tests/integration/test_train_resume_cpu.py::test_resume_policy_warn_allows_
model_fingerprint_mismatch`, but production configs use `strict`). `resume_
from` is `null` in every shipped config — an operator resumes an interrupted
run by pointing it at the last same-run checkpoint, e.g. `training/cache/
coding/checkpoints/checkpoint-200`, never at another capability's adapter.

This was independently verified as reliable, not just configured, via the
existing test suite (unmodified by this phase, all passing):
`tests/integration/test_train_resume_cpu.py` (interrupted-training recovery,
CPU path) and `tests/golden/test_golden_resume_equivalence.py` (a resumed run
must produce byte-identical results to an uninterrupted one — the strongest
possible resume-reliability guarantee, and it is CI-enforced, not just
smoke-tested).

`context` (50,161 records, the largest corpus by ~9×) is the one capability
where multi-session resume is operationally likely, not just a theoretical
safety net — its estimated 10–20 h wall-clock (§2) can plausibly exceed one
Colab session. The shared `save_steps: 200` / `save_total_limit: 3` /
`resume_from` mechanism already covers this without any capability-specific
configuration; no change was needed or made.

## 7. Validation handoff

Every published Capability Package's `artifact.json` (built by
`aiodoo_training.artifacts.publish_contract.build_adapter_artifact_json`) is
consumed downstream in two places, both verified field-for-field against
what training actually writes:

- **`aiodoo-validation`** resolves and certifies an adapter by
  `capability_id` + `adapter_type` (frozen requirement: `adapter_type` must
  equal the capability id / validation profile name — see `publish_contract.
  py`'s module docstring). Per-capability validation profiles exist for all 7
  Capability Contract members (`coding`, `planner`, `repair`, `execution`,
  `conversation`, `approval`, `evaluation` — verified present under
  `aiodoo-validation/aiodoo_validation/capabilities/`); `context` has none,
  because it is not a Capability Contract member (§10) — an adapter trained
  from `context` data is not independently certified by `aiodoo-validation`,
  by design.
- **Request-generation gold corpora**: `aiodoo-datasets`' per-capability
  `<capability>_eval_corpus.jsonl` files (built via `generators/common/
  contract/eval_corpus.py` from the same canonical `aiodoo_contract`
  request/response schemas `aiodoo-validation` itself uses) are exactly the
  `(request, expected_response)` gold pairs a validation run certifies an
  adapter against. These exist today for all 7 contract-capability datasets
  and are internally contract-valid (each record is checked with `aiodoo_
  contract.validators.ContractValidator` before being written) — see §10 for
  a known, narrow false-positive in `aiodoo-datasets`' own *internal*
  quality-gate re-check of these files (not a defect in the corpora
  themselves, and not something that blocks a validation run from reading
  them).

No manual intervention step exists or is required between a training run's
publish step and a validation run picking up the same `capability_id`.

## 8. Packaging handoff

Every field `aiodoo_model.publishing.normalize._map_fields` reads from a
training `artifact.json` was cross-checked against what `build_adapter_
artifact_json`/`build_merged_artifact_json` actually write, one field at a
time: `artifact_type`/`artifact_kind` → `_resolve_kind`; `capability_id`
(required for adapters) → present; `supported_odoo_versions` (required for
adapters) → present, defaulting to `(17, 18, 19)`; `protocol_major` (read as
`supported_protocol_major` fallback) → present as `protocol_major`;
`peft_type` → present; `base_artifact_id`/`base_model_ref` (merged only, not
applicable to Stage-1 independent adapters) → n/a by design; `contract_
version` (fail-closed `ensure_contract_compatible` check on ingest) →
present, pinned to the installed `aiodoo_contract.version.CONTRACT_VERSION`;
`capability_package_metadata` (the canonical `CapabilityPackageMetadata`
block) → present whenever `capability_id` is a real `CapabilityName` member
and family/architecture are resolvable (correctly `None`, never fabricated,
for `context`). Every required field lines up; no format conversion step
exists or is needed between a training publish and a model-registry publish
call.

Storage layout: `directory_naming` in each capability's `export.yaml`
consistently follows `models/adapters/aiodoo-<capability>/<adapter_name>/`,
`models/merged/aiodoo-<capability>/`, `models/exports/aiodoo-<capability>/`,
matching the workspace layout `aiodoo-model`'s registry and `aiodoo-colab`'s
`ModelStore`/Drive layout both already expect (verified unchanged from Phase
8/9 audits — no registry or storage-layout change was made or is needed).

## 9. Artifact lifecycle

```text
Checkpoint (training/cache/<cap>/checkpoints/checkpoint-N/)
   │  save_steps=200, save_total_limit=3, validate_on_load=true
   ▼
Publish preflight (validate_checkpoint_for_publish: adapter_config.json +
   adapter weights required; checkpoint sidecars stripped)
   ▼
Capability Package (models/adapters/aiodoo-<cap>/)
   = peft_adapter + tokenizer + manifest + model_card + bundle (export_types)
   + artifact.json (publish_contract.py — the validation/packaging handoff shape)
   │  atomic_replace_directory: tmp-publish dir verified, then renamed into
   │  place — a failed publish never leaves a partial adapter directory
   ▼
aiodoo-validation certification (request/response eval corpus, §7)
   ▼
aiodoo-model registry publish (Stage 2: normalize → validate → resolve, §8)
   ▼
[out of scope for this phase: promotion / merge / Personal AIODOO packaging]
```

Every stage in this lifecycle up to (and including) the `aiodoo-model`
registry publish call was verified to have matching, non-duplicated field
contracts on both sides of each arrow in this phase; no stage was modified.

## 10. Known limitations

1. **Three of eight capability datasets are placeholders, not training-scale
   corpora** (`conversation`, `approval`, `evaluation` — 1 record each). This
   is a pre-existing, explicitly documented `aiodoo-datasets` limitation
   (`docs/production_freeze_report.md`, `docs/FUTURE_INTEGRATION_
   IMPROVEMENTS.md`), not something this phase introduced, hid, or was asked
   to fix (`aiodoo-datasets` is frozen; "no regeneration unless genuinely
   required" — regenerating richer corpora is exactly the kind of dataset
   redesign this phase is explicitly told not to do). Their training configs
   are fully valid and would work correctly the moment richer datasets are
   published; nothing in `aiodoo-training` blocks that.

2. **`context` is a training product, not a Capability Contract member.**
   It produces a real, trainable, publishable adapter (`aiodoo-context`) with
   a full config pack like every other row in the matrix, but it has no
   `aiodoo-validation` certification profile and no request/response shape
   in `aiodoo_contract.schemas.enums.CapabilityName` — by design, not by
   omission (`context` is retrieval-result infrastructure other capabilities
   consume, not itself an invokable capability). This is already correctly
   reflected everywhere it matters: `aiodoo-training/CONTRACT_ADOPTION.md`
   §7, `aiodoo_training/artifacts/publish_contract.py`'s `_capability_
   package_metadata` (returns `None` rather than fabricating a contract
   block for `context`), and confirmed by this phase's audit of
   `aiodoo-validation`'s capability directory (7 profiles, no `context`).

3. **`aiodoo-datasets`' own internal validation framework mis-classifies its
   newer per-capability `<capability>_eval_corpus.jsonl` files** (added by a
   prior, already-merged change closing `ACT-007`/`DEF-05`). Root cause: the
   shared `SchemaRegistry._infer_generator` (`validation/schemas/registry.
   py`) infers a dataset's schema by a case-insensitive substring match on
   the filename (e.g. any name containing `"approval"` resolves to the
   `approval-v1` training-record schema), so `approval_eval_corpus.jsonl` —
   whose actual record shape is the *contract* shape
   (`{capability, request, expected_response, source_protocol_hash}`, not
   the training-record shape) — is checked against the wrong schema and
   reports spurious `SCH-001` "missing required field" failures. Evidence
   this is a false positive, not a real defect: `generators/common/contract/
   eval_corpus.py::build_eval_corpus` already validates every record with
   `aiodoo_contract.validators.ContractValidator` **before** writing it, so
   the files on disk are contract-valid; it is only the dataset repo's own
   *second*, unrelated legacy schema check that misfires on them.
   **Not fixed by this phase**: `aiodoo-datasets` is frozen, this heuristic
   predates and is unrelated to production training readiness (training
   never reads `*_eval_corpus.jsonl` files — only `aiodoo-validation` does,
   and it uses its own contract-aware corpus loader, not this heuristic),
   and correcting a shared, hash-fingerprinted schema-registry module is a
   scope decision for whoever owns `aiodoo-datasets` next, not a training
   readiness blocker. Recorded here so it is not mistaken for something this
   phase missed.

4. **GPU/duration estimates in §2 are planning-grade, not measured.** No
   actual Qwen3-8B QLoRA run was performed on real GPU hardware during this
   phase (out of scope — "Do NOT begin production adapter training"); the
   existing golden/integration test suite exercises the same code paths on
   CPU with tiny fixtures for correctness and determinism, not for
   wall-clock or VRAM measurement.

5. **Deferred protocol surface** (`RejectPlan`/`SendAnswer` not yet exposed
   as `aiodoo-core` V1 wire commands) and **cosmetic `ruff format` drift** in
   a handful of files across `aiodoo-contract`/`aiodoo-validation`/
   `aiodoo-model` remain exactly as documented in the Phase 9
   `ECOSYSTEM_CERTIFICATION.md` §11 — unrelated to production training
   readiness, re-confirmed still accurate, not re-litigated here.
