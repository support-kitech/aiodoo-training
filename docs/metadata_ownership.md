# Metadata Ownership

**Status:** Authoritative metadata specification  
**Binding:** [ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md)  
**Related:** [Lifecycle](lifecycle.md), [Terminology](terminology.md)

---

## Principle

There is **not** one unified metadata schema across the ecosystem.

Multiple metadata models exist **intentionally**. Each document has a different
producer, consumer, and lifecycle. Unifying them into a single DTO would couple
frozen repositories and force breaking churn.

Training produces **producer documents**.  
`aiodoo-model` owns **normalized registry metadata** after registry publish.  
`aiodoo-validation` owns **evidence / report** documents.

---

## Metadata map

| Document | Produced by | Consumed by | Purpose |
|----------|-------------|-------------|---------|
| Checkpoint sidecars (`rng.json`, `metrics.json`, …) | Training | Training resume | Internal continuity |
| Adapter Drive `manifest.json` | Training Drive publish | Training / operators | Diagnostics (not registry contract) |
| **`artifact.json`** (Capability Package / base cache) | Training | Validation + Model normalize | **External handoff metadata** |
| **`export_manifest.json`** (ArtifactBundle) | Training export | Export integrity / inventory | Bundle contract |
| Bundle checksums | Training export | Integrity verify | Bundle / optional package verify |
| Evaluation / quality JSON sidecars | Training evaluation | Operators; optional bundle roles | Training-local quality |
| Experiment `summary.json`, metrics, tracking JSONL | Training tracking | Operators / CI | Run lineage |
| Validation reports / certification evidence | Validation | Humans; model stores **refs only** | Certification |
| Registry MetadataDocument | Model normalize | Registry / compatibility / promotion | Canonical registered metadata |
| Release / channel metadata | Model | Consumers | Product/release identity |

---

## Why separation is correct

| Risk of one schema | Avoided by separation |
|--------------------|------------------------|
| Training forced to import model types | On-disk JSON only |
| Validation forced to understand export roles | Uses `artifact.json` resolution |
| Registry forced to store training sidecars | Normalize + fingerprint on ingest |
| Every optional field becomes a cross-repo break | Additive optional fields per document |

---

## Training obligations (conceptual)

Capability Package `artifact.json` should be rich enough for consumers. Frozen
`aiodoo-model` normalize expects adapters to resolve capability identity,
protocol major, family/architecture, and supported Odoo versions (from the
package and/or `PublishingRequest`). Enrichment of producer fields is an
**implementation** phase (B1+), not a B0 behavior change.

### Capability Package `artifact.json` field roles (B1 / Option A)

| Field | Role |
|-------|------|
| `artifact_type` | Frozen protocol kind (`coding_adapter` / `base_model` / `merged_model`) |
| `capability_id` | Business skill identity |
| `adapter_type` | Skill label required by frozen `aiodoo-validation` profiles (equals capability id). **Retained — do not remove.** |
| `peft_type` | PEFT implementation (`lora` / `qlora`) for model normalize |

Until packages are fully self-describing, callers may still supply missing fields via
model `PublishingRequest`. The architectural target remains **self-describing
Capability Packages**.
---

## Non-goals

- Sharing Python metadata classes across repositories
- Embedding full validation trees into training packages
- Making ArtifactBundle manifest the registry metadata document
