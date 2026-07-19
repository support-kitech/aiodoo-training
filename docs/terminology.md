# Terminology Glossary

**Status:** Authoritative vocabulary for `aiodoo-training`  
**Binding:** [ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md)  
**Related:** [Ownership](ownership.md), [Capability Model](capability_model.md), [Product Model](product_model.md)

This glossary is the single source of truth for public documentation language.
If another document conflicts with this file, **this file wins** (then amend the
other document).

---

## Core identities

| Term | Definition | Not to be confused with |
|------|------------|-------------------------|
| **Capability** | A first-class skill identity trained as one independent adapter from a base model. Public ids: `coding`, `repair`, `execution`, `planner`, `context`, `conversation`, `approval`, `evaluation`. | **Product**; `TrackingCapability` (tracker backend feature flags) |
| **Capability ID** | The public training id string (same as catalog id under `configs/training/<id>/`). | Legacy internal `EXP-NNNN` bookkeeping ids |
| **Product** | A composed offering for consumers: **Development** or **Reasoning**. Owned by `aiodoo-model`. | Capability adapters; Drive folder names like `aiodoo-coding` |
| **Training ID** | Public capability id used in configs, cache paths, and experiment folders. | Product name |

---

## Packages and artifacts

| Term | Definition | Authority |
|------|------------|-----------|
| **Checkpoint** | Training-local weight tree plus sidecars for resume. | **Internal** |
| **Capability Package** | On-disk inference package for one capability (typically `models/adapters/aiodoo-<id>/` or `models/merged/aiodoo-<id>/`) including root **`artifact.json`**. | **Authoritative external handoff** |
| **ArtifactBundle** | Portable export directory with `export_manifest.json`, checksums, and role trees under `artifacts/`. Produced by `ExportManager`. | **Export inventory** (retained; not the primary registry ingest shape) |
| **Artifact** | Generic file or logical role inside a package (weights, tokenizer, card, report). Prefer the specific package term in architecture docs. | — |
| **Export** | Training stage that materialises an ArtifactBundle (and related inventory). | Training |

---

## Publish and consume verbs

| Term | Definition | Owner |
|------|------------|-------|
| **Drive publish** | Writing Capability Packages (and related experiment trees) into the workspace layout. | Training (`ArtifactOutputManager`) |
| **Registry publish** | Ingesting a Capability Package into the model registry. | `aiodoo-model` (`PublishingService`) |
| **Publish** (unqualified) | Ambiguous — **do not use** in new docs without Drive or Registry qualifier. | — |

---

## Evaluation and quality

| Term | Definition | Owner |
|------|------------|-------|
| **Training evaluation** | Offline metrics and quality gates inside training (`evaluation/`). | Training |
| **Validation** | Capability certification pipelines in `aiodoo-validation`. | Validation |
| **Certification** | Formal validation outcome / evidence produced by validation. | Validation |
| **Quality gate** | Training-local pass/fail policy that may block export. | Training |

---

## Model lifecycle (ecosystem)

| Term | Definition | Owner |
|------|------------|-------|
| **Registry** | Immutable artifact catalog. | `aiodoo-model` |
| **Release** | Immutable binding of registered artifacts for a channel/version. | `aiodoo-model` |
| **Promotion** | Moving a release across channels under policy + evidence. | `aiodoo-model` |
| **Compatibility** | Policy evaluation of artifact/release vs consumer requirements. | `aiodoo-model` |
| **Deployment** | Delivering resolved artifacts to an environment. | Outside training (ops / runtime) |
| **Runtime** | Serving / agent execution that loads models. | `aiodoo-core` / runtime stacks — **not** training |

---

## Repository names

| Correct | Incorrect / legacy |
|---------|-------------------|
| `aiodoo-model` | `aiodoo-models` |
| `aiodoo-training` | — |
| `aiodoo-validation` | — |
| `aiodoo-datasets` | — |
| `aiodoo-core` | — |

---

## ADR citation

| Citation | Meaning |
|----------|---------|
| **Ecosystem ADR-0001 — AIODOO Model Lifecycle** | Cross-repository lifecycle (being finalized / finalized outside this repo’s `docs/adr/0001-*`). |
| **ADR-0001** (in this repo) | [Immutable Domain](adr/0001-immutable-domain.md) only. |
| **ADR-0022** | [Package Surfaces & Lifecycle Alignment](adr/0022-package-surfaces-lifecycle-alignment.md). |

---

## Deprecated phrases (do not use in new writing)

| Deprecated | Replace with |
|------------|--------------|
| product adapter | capability adapter / Capability Package |
| aiodoo-models | aiodoo-model |
| ArtifactBundle is the only Models interface | Capability Package is the authoritative external handoff; ArtifactBundle remains the export inventory |
| publish (alone) | Drive publish / registry publish |
| validation (for training metrics) | training evaluation |
