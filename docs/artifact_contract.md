# AIODOO Artifact Contract

**Status:** Official **ArtifactBundle** export contract (**Phase 4 permanently frozen**)  
**Clarified by:** [ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md) (2026-07-19)  
**Binding to:** Phase 4 Evaluation & Export (ADR-0015)  
**Protocol:** `artifact_protocol_version = "1"`  
**Related:** [Lifecycle](lifecycle.md), [Artifact Output Pipeline](artifact_output_pipeline.md), [Phase 4 Architecture](phase4-evaluation-export-architecture.md), [ADR-0015](adr/0015-phase4-evaluation-export.md), [Frozen Public Contracts](frozen_public_contracts.md), [Terminology](terminology.md)

This document is the **authoritative on-disk contract for `ArtifactBundle`**
(export inventory). It describes portable files and JSON only. It does **not**
redesign Phase 0–3 frozen surfaces and does **not** introduce runtime Python
coupling between repositories.

---

## Clarification (ADR-0022) — dual package surfaces

Phase 4 historically described `ArtifactBundle` as the sole Training → models
interface. Production and frozen `aiodoo-model` registry publish instead ingest
**Capability Packages** (directories with root `artifact.json`).

| Surface | Document | Authority |
|---------|----------|-----------|
| **Capability Package** | [Lifecycle](lifecycle.md), [Artifact Output Pipeline](artifact_output_pipeline.md) | **Authoritative external handoff** to `aiodoo-validation` and `aiodoo-model` registry publish |
| **ArtifactBundle** | **This document** | **Authoritative export inventory** (roles, fingerprints, optional merged tree) |

Both surfaces remain. Neither replaces the other. See [ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md).

```text
aiodoo-datasets → aiodoo-training → Capability Package → aiodoo-validation / aiodoo-model
                                 ↘ ArtifactBundle (export inventory; optional merged source)
```

---

## Purpose

`ArtifactBundle` is the supported **portable export package** produced by
`ExportManager`. Frozen `aiodoo-model` **registry publish** consumes Capability
Packages (`artifact.json` layout), not this bundle root, unless a future
consumer explicitly loads Bundle roles.

| Repository | Responsibility |
|------------|----------------|
| **AIODOO Training** | Trains, evaluates (offline), **exports** `ArtifactBundle`, and **Drive-publishes** Capability Packages |
| **aiodoo-model** | **Registry-publishes** Capability Packages; resolves/loads for consumers |
| **aiodoo-validation** | Certifies Capability Packages |

Neither repository imports runtime objects from the other:

- `aiodoo-model` **must not** import `aiodoo_training` (domain, ports, or infrastructure).
- Training **must not** import `aiodoo_model`.
- Live objects such as `TrainerBackend`, `CheckpointManager`, `TrainingSession`,
  `EvaluationSession`, `ExportSession`, PEFT/Torch training types, or framework
  carriers **never** cross the boundary.
- Sidecar evaluation / quality documents are **optional portable JSON**, not live
  Training class instances.

The Bundle contract is validated by **files + checksums + `artifact_protocol_version`**,
not by shared Python packages.

---

## Artifact Bundle Layout

An `ArtifactBundle` is a single directory published atomically by Training’s
`ExportManager`. Bundle consumers open an explicit bundle path and read the layout below.

```text
ArtifactBundle/
  export_manifest.json      # ExportManifest — Bundle inventory contract surface
  checksums.sha256          # sha256 digests of content files
  artifacts/
    adapter/                # PEFT adapter files (when peft_adapter exported)
    tokenizer/              # tokenizer files (when tokenizer exported)
    merged/                 # optional merged weights
    …                       # additional role trees as export_types expand
  model_card.md             # human-readable card (when model_card exported)
  model_card.json           # optional structured card
  evaluation/
    report.json             # optional EvaluationReport JSON DTO
    quality_report.json     # optional QualityReport JSON DTO
```

**Layout rules**

1. Paths inside the manifest are **posix relative** to the bundle root.
   Absolute host paths must never appear in the Bundle-facing contract.
2. `export_manifest.json` is always present for a valid published bundle.
3. Content files under `artifacts/` are materialised by an `Exporter` adapter;
   Training’s `ExportManager` owns inventory, checksums, fingerprint, index
   update, and atomic publish (`tmp` → `rename`).
4. Directory name form used by Training (informative, not a load gate):
   `bundle-<experiment_id>-<export_fingerprint_prefix>/`.

Partial publishes are forbidden: on failure the temporary tree is discarded and
any prior published bundle remains intact.

---

## ExportManifest

`export_manifest.json` is the **authoritative per-bundle inventory**. Bundle consumers
parse it first. Field names and semantics for protocol `"1"`:

| Field | Required for Bundle load | Meaning |
|-------|--------------------------|---------|
| `schema_version` | Yes (to parse) | Manifest JSON shape version (currently `"1"`) |
| `artifact_protocol_version` | **Yes** | Sole semantic version for package layout and required roles |
| `experiment_id` | Yes (identity) | Experiment identity string |
| `run_id` | Yes (identity) | Run identity string |
| `model_fingerprint` | Yes (FULL loads) | Portable base-model identity digest |
| `adapter_fingerprint` | Yes (FULL loads) | Portable adapter identity digest |
| `config_fingerprint` | Yes (FULL loads) | Portable config identity digest |
| `evaluation_fingerprint` | Optional | Digest when evaluation is attached; omit / null if none |
| `export_backend_key` | **Ignored for load** | Training diagnostic (e.g. `stub`, `hf_peft`) |
| `export_types` | Yes (discovery) | Declared export roles / types for this bundle |
| `artifacts` | **Yes** | Tuple/list of `ArtifactDescriptor` entries |
| `required_artifacts` | Yes (integrity) | Relative paths that must exist |
| `artifact_paths` | Yes (integrity) | Flat list of all relative paths (must match `artifacts`) |
| `export_fingerprint` | Yes (identity) | Bundle identity digest (see Determinism) |
| `training_protocol_version` | **Ignored for load** | Training resume-protocol echo / provenance only |
| `software` | **Ignored for load** | Diagnostic map (`python`, `aiodoo-training`, …) |
| `created_at` | Optional / non-golden | ISO timestamp; not part of deterministic identity |

### `artifact_protocol_version`

- Current produce value: **`"1"`**.
- This is the **only** semantic version Bundle consumers use as a load gate for layout and
  required roles.
- Bump when layout or required roles change in a breaking way.
- Do **not** use `training_protocol_version` as a Bundle load gate.

### Versioning

| Version field | Owner | Purpose |
|---------------|-------|---------|
| `artifact_protocol_version` | Training produces; Bundle consumer consumes | Package layout + role contract |
| `schema_version` | Manifest JSON | Parseability of the manifest document |
| `training_protocol_version` | Training echo | Provenance only |

**Migration rule:** Bundle consumers may support N and N−1 protocols via its own
`ArtifactCompatibilityPolicy`. Training always writes the current produce
version. Adapters for older bundles live in the consumer, not by rewriting
Training’s frozen contracts.

### Required fields (load vs diagnostics)

**Must be present and valid for a defensive Bundle load**

- `schema_version`
- `artifact_protocol_version`
- `artifacts` (and consistent `required_artifacts` / `artifact_paths`)
- fingerprints needed for the chosen load mode (model / adapter / config)
- `export_types` / roles sufficient to select loaders

**Must not cause hard-fail solely because they are missing or unfamiliar**

- `training_protocol_version`
- `export_backend_key`
- `software.*`

---

## ArtifactDescriptor

Each entry in `ExportManifest.artifacts` describes one logical file (or card
sidecar) inside the bundle:

| Field | Meaning |
|-------|---------|
| `role` | Logical role string (e.g. `peft_adapter`, `tokenizer`, `model_card`, `manifest`, `checksums`, `evaluation_report`) |
| `relative_path` | Posix path relative to bundle root |
| `checksum` | SHA-256 hex digest of file contents |
| `content_type` | Optional MIME / kind hint |
| `required` | If `true`, integrity validation requires the file to exist |

**Rules**

1. `ExportManifest.artifacts` is the authoritative inventory. Operators must not
   rely on ad-hoc filesystem walks for discoverability.
2. `required_artifacts` / `artifact_paths` stay consistent with `artifacts`.
3. Checksums are content hashes of individual files. They feed into export
   fingerprint computation (see Determinism).
4. There is **no separate per-descriptor fingerprint field** in protocol `"1"`.
   Bundle-level identity is `export_fingerprint`; file identity is `checksum`.

---

## ArtifactIndex

Training maintains a **repository-wide** (export-root) catalog, typically:

```text
<output_dir>/artifacts.json
```

Each `ArtifactIndexEntry` records:

| Field | Meaning |
|-------|---------|
| `bundle_path` | Path relative to the export output root |
| `experiment_id` / `run_id` | Bundle identity |
| `export_fingerprint` | Published fingerprint |
| `artifact_protocol_version` | Protocol of that bundle |
| `export_types` / `roles` | Discovery metadata from the manifest |
| `created_at` | Optional |
| `manifest_relpath` | Default `export_manifest.json` |

**Purpose:** answer “which bundles exist under this export root?” without
opening every tree. Update happens **after** atomic publish.

**Bundle consumers do not require ArtifactIndex.**

- Consumers open an **explicit bundle directory**.
- Load requires only that bundle’s `ExportManifest` + files.
- Index is a Training / operator / CI convenience. Absence of Index at a remote
  Bundle deployment is not a load failure.

**Invariant:** every published bundle appears in the Index when Training updates
it; every role inside a bundle appears in that bundle’s `ExportManifest.artifacts`.

---

## Artifact Validation

`ArtifactValidationPolicy` answers: **Is this single bundle internally sound?**

It is a **producer-side integrity** policy (and a semantic mirror for consumers’
own integrity checks). It is **not** the consumer protocol matrix.

| Policy | Behaviour |
|--------|-----------|
| **STRICT** | Required files present or reject; checksum mismatches reject; fingerprint rematch reject; unknown/`artifact_protocol_version` problems reject; quality gates reject when `require_pass_for_export` |
| **WARN** | Required files still reject; checksum / fingerprint / gate issues may warn; protocol still must be known |
| **RELAXED** | Required files and known protocol still enforced at produce; checksum / fingerprint / gate failures may degrade to warn or ignore per Training policy |

Typical Training STRICT expectations:

| Check | STRICT |
|-------|--------|
| Required files present | reject |
| Checksums match | reject |
| Fingerprints present / rematch | reject |
| `artifact_protocol_version` writable/known | reject |
| Quality gates when `require_pass_for_export` | reject |
| Software package versions | ignore |

Consumers **must still validate defensively** on consume, even when Training
published under STRICT. Integrity code is **not** shared as a Python dependency.

---

## Artifact Compatibility

`ArtifactCompatibilityPolicy` answers: **May this protocol and role set be
consumed by a given consumer runtime profile?**

| Concern | Policy |
|---------|--------|
| Internal soundness (files, checksums, fingerprints) | `ArtifactValidationPolicy` |
| Consumer protocol / role negotiation | `ArtifactCompatibilityPolicy` |

Conceptual fields:

```text
ArtifactCompatibilityPolicy
  accepted_artifact_protocols: tuple[str, ...]   # e.g. ("1",)
  required_roles: tuple[str, ...]                # e.g. ("peft_adapter", "manifest")
  optional_roles: tuple[str, ...]
  reject_unknown_roles: bool                     # default false — forward compatible
```

### Compatibility negotiation

1. Training writes `artifact_protocol_version` and declared roles.
2. Training may optionally **preflight** against a configured target profile
   (warn if it would emit a protocol a named consumer profile cannot read).
   Training never imports consumer code.
3. The consumer owns the **authoritative** consumer matrix and rejects unsupported
   protocols/roles without consulting Training.

### Protocol evolution

- Additive optional roles (new export types) may stay on the same protocol when
  older consumers can ignore unknown roles (`reject_unknown_roles=false`).
- Breaking layout or newly required roles → bump `artifact_protocol_version`.
- Older-bundle adapters live in the consumer; Training continues writing the current
  produce version.

Integrity alone cannot express multi-year independent consumer releases — hence
the separate compatibility abstraction.

---

## Determinism

Given the same portable inputs:

- same model  
- same adapter  
- same tokenizer materialisation  
- same evaluation dataset (when evaluation is included)  
- same configuration  
- same seed  
- same resolved `ExecutionEnvironment`  

Training guarantees that a re-export produces the **same**:

- `export_fingerprint`
- `ArtifactDescriptor` relative paths and content checksums (content files)
- portable `ExportManifest` identity fields
- model-card metadata used for fingerprinting  

**excluding** timestamps and explicitly non-deterministic diagnostics such as:

- `created_at`
- volatile `software.*` host details where environments differ
- self-referential checksum churn of the manifest file itself when the write
  order embeds the manifest into its own inventory (content file checksums and
  the export fingerprint remain the stable golden surface)

**Fingerprint rules**

- Absolute output paths **must never** enter fingerprint material.
- Export fingerprint digests protocol, fingerprints, export types, model-card
  digest, and descriptor `relative_path:checksum` pairs in a canonical order.
- JSON used for digests uses sorted keys / stable serializers.

CPU golden tests in Training lock this invariant for stub backends.

---

## Repository Boundary

```text
Training exports     →  ArtifactBundle (export inventory)
Training Drive publish →  Capability Package → validation / aiodoo-model registry publish
```

| Concern | Training | aiodoo-model |
|---------|----------|--------------|
| Train / resume / checkpoint | ✓ | ✗ |
| Offline evaluation for quality gates | ✓ | optional smoke only |
| Build ArtifactBundle | ✓ | ✗ (does not require Bundle for registry publish) |
| Drive-publish Capability Package | ✓ | ✗ (consumes via registry publish) |
| Validate package integrity at produce | ✓ | ✓ (defensive consume) |
| Compatibility matrix for runtimes | optional preflight | **authoritative** |
| Inference / serving / batch generate | ✗ | load plans; serving elsewhere |
| vLLM / SGLang / serving stacks | ✗ | ✗ (runtime) |
| Runtime-only quantization for serving | ✗ | ✗ (runtime) |

**Nothing else crosses the boundary:**

- No shared live sessions or managers.
- No framework types (Torch / Transformers / PEFT / etc.) in the handoff.
- No requirement that `aiodoo-model` understand Training resume protocol.
- No requirement that `aiodoo-model` read `ArtifactIndex`.

---

## Future Extensions

New formats and runtimes **extend** this contract by registration and optional
roles — they do not invent a parallel handoff.

| Extension | Fit |
|-----------|-----|
| **GGUF** | Additive `ExportType` / exporter registry key; new `artifacts/` role tree + descriptors; same `ExportManifest` |
| **ONNX** | Same pattern: role + files + checksums under protocol `"1"` or a bumped protocol if required roles change |
| **TensorRT** | Engine artefacts as optional roles; Consumer engine loaders consume roles they understand |
| **vLLM** | **Runtime** profile: may select roles from a Bundle or load registry-resolved artifacts; no Training redesign |
| **SGLang** | Same as vLLM — consumer-side runtime over on-disk contracts |

Rules for extensions:

1. Prefer additive optional roles with forward-compatible consumers.
2. Breaking required-role or layout changes bump `artifact_protocol_version`.
3. Serving / inference engines belong to **runtime**; Training only packages
   portable artefacts consumers can open.
4. Never widen frozen Training port signatures solely to support a new runtime.
5. Registry publish extensions must preserve Capability Package `artifact.json`
   compatibility with frozen `aiodoo-model` normalize rules.

---

## Validation checklist (ArtifactBundle consume path)

For consumers that open an **ArtifactBundle** (not registry publish):

1. Parse `export_manifest.json` (`schema_version`).
2. Accept `artifact_protocol_version` via `ArtifactCompatibilityPolicy`.
3. Verify required roles / files and content checksums (integrity).
4. Verify fingerprints needed for the load mode.
5. Optionally read `evaluation/*` JSON for tags — without importing Training.

For **registry publish**, see Capability Package layout in
[artifact_output_pipeline.md](artifact_output_pipeline.md) and frozen
`aiodoo-model` publishing docs — start from root `artifact.json`.

---

## Document authority

This file is the **official Artifact Contract** referenced by Phase 4
architecture and ADR-0015. Implementation details live under
`aiodoo_training/domain/export_manifest.py` and `aiodoo_training/export/`.
If implementation and this document ever diverge, **stop**, explain the gap, and
resolve via the Section 9 change process in `frozen_public_contracts.md` —
do not silently invent a second handoff path.
