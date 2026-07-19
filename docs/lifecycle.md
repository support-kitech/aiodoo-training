# Training & Artifact Lifecycle

**Status:** Authoritative lifecycle specification  
**Binding:** [ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md), Ecosystem ADR-0001 — AIODOO Model Lifecycle  
**Related:** [Ownership](ownership.md), [Artifact Contract](artifact_contract.md), [Artifact Output Pipeline](artifact_output_pipeline.md)

This document describes the **canonical lifecycle** as operated by
`aiodoo-training` and handed off to frozen siblings. It corrects idealized
single-chain diagrams: **Capability Package and ArtifactBundle are siblings**.

---

## End-to-end lifecycle

```text
Dataset (aiodoo-datasets)
        ↓
Capability training (aiodoo-training)
        ↓
Checkpoint (internal)
        ├──→ Training evaluation (optional quality gates)
        │
        ├──→ Capability Package          [Drive publish — authoritative external]
        │         ↓
        │    Validation (aiodoo-validation)
        │         ↓
        │    Certification evidence
        │         ↓
        │    Registry publish (aiodoo-model)
        │         ↓
        │    Release / promotion (aiodoo-model)
        │         ↓
        │    Deployment / Runtime (outside training)
        │
        └──→ ArtifactBundle              [ExportManager — export inventory]
                  ↓
             optional Merged Capability Package (from bundle merged role)
                  ↓
             (same validation / registry path when published)
```

---

## Stage ownership

| Stage | Owner | Notes |
|-------|-------|-------|
| Dataset preparation | `aiodoo-datasets` | Training only consumes |
| Training | `aiodoo-training` | One capability per run |
| Checkpoint | `aiodoo-training` | Internal; not a handoff package |
| Training evaluation | `aiodoo-training` | Not certification |
| Capability Package (Drive publish) | `aiodoo-training` | External handoff |
| ArtifactBundle export | `aiodoo-training` | Inventory / merged source |
| Capability validation / certification | `aiodoo-validation` | |
| Registry publish / release / promotion | `aiodoo-model` | |
| Product composition | `aiodoo-model` | After capabilities exist |
| Deployment / runtime | Outside training | |

---

## Package surfaces (summary)

| Package | Purpose | Owner | Consumer | Canonical? |
|---------|---------|-------|----------|------------|
| Checkpoint | Resume | Training | Training | Internal |
| Capability Package | External handoff | Training | Validation, Model | **Yes (external)** |
| ArtifactBundle | Portable export inventory | Training | Operators; merged extract; future role loaders | Yes (export) |
| Merged Capability Package | Optional full weights | Training | Validation / Model when used | Optional external |

### Authority rules

1. **External** (validation + registry publish): Capability Package with root
   `artifact.json` is authoritative. Matches frozen `aiodoo-model` publishing
   ingest layout.
2. **Export inventory**: ArtifactBundle with `export_manifest.json` remains the
   Phase 4 portable bundle contract.
3. Do not claim either surface is the “only” package in the repository; claim
   the correct authority per consumer.

---

## Training run lifecycle (within this repo)

```text
Resolve config + workspace
    → Load dataset / tokenize / pack
    → Load base model + adapt (LoRA/QLoRA)
    → Train / resume
    → Save checkpoints
    → Evaluate (if enabled)
    → Export ArtifactBundle (if enabled)
    → Finalize Drive publish (Capability Package ± merged)
    → Experiment summary / tracking close
```

Same-run resume restores from checkpoints. Cross-capability resume is forbidden.

---

## Handoff contracts

| From → To | Contract | Location |
|-----------|----------|----------|
| Training → Validation | Capability Package (`artifact.json` + weights) | Drive adapter/merged dirs; base model cache `artifact.json` |
| Training → Model (registry publish) | Same Capability Package shape | Model normalize accepts training `artifact.json` |
| Training → Runtime | None directly | Runtime consumes model resolve/load |
| Training → Operators/CI | ArtifactBundle + experiment trees | Export + `experiments/<id>/` |

---

## Non-goals

- Implementing certification inside training
- Implementing registry publish inside training
- Emitting Development / Reasoning product packages
