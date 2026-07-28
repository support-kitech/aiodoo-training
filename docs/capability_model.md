# Capability Model

**Status:** Authoritative capability specification for `aiodoo-training`  
**Binding:** [ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md)  
**Related:** [Product Model](product_model.md), [Terminology](terminology.md), Ecosystem ADR-0001 — AIODOO Model Lifecycle

---

## Definition

A **Capability** is a first-class skill that AIODOO trains as **one independent
adapter** from a configured **base model** (fresh LoRA/QLoRA). Capabilities are
not products. Products are composed later in `aiodoo-model`.

```text
Capability ≠ Product
```

---

## Capability identity

| Field | Rule |
|-------|------|
| Public id | One of the catalog training ids |
| Config root | `configs/training/<id>/` |
| Adapter directory name | `aiodoo-<id>` (Drive path label; not a Product) |
| Package metadata | Must eventually carry `capability_id` aligned with this id (implementation phase) |

### Catalog

| Capability ID | Published adapter dir | Dataset (v1.0.0) | Notes |
|---------------|----------------------|------------------|-------|
| `coding` | `aiodoo-coding` | `coding_v1_0.jsonl` | Development skill |
| `repair` | `aiodoo-repair` | `repair_v1_0.jsonl` | Development skill |
| `execution` | `aiodoo-execution` | `execution_dataset.jsonl` | Development skill |
| `planner` | `aiodoo-planner` | `planner_v1_0.jsonl` | Reasoning skill |
| `context` | `aiodoo-context` | `context_v1_0.jsonl` | **Independent capability** (see below) |
| `conversation` | `aiodoo-conversation` | `conversation_dataset.jsonl` | Reasoning skill |
| `approval` | `aiodoo-approval` | `approval_dataset.jsonl` | Reasoning skill |
| `evaluation` | `aiodoo-evaluation` | `evaluation_dataset.jsonl` | Reasoning skill — judgment SFT (v2); not BenchmarkCatalog |

Catalog stage numbers in pack READMEs are **labels only**. They do **not** imply
weight chaining. `checkpointing.resume_from` is same-run recovery only.

---

## Ownership

| Concern | Owner |
|---------|-------|
| Capability training | `aiodoo-training` |
| Capability corpora / schemas | `aiodoo-datasets` |
| Capability certification packs | `aiodoo-validation` |
| Capability registration metadata | `aiodoo-model` (after registry publish) |
| Runtime invocation of a skill | `aiodoo-core` / runtime (consumes resolved artifacts) |

---

## Capability package

A **Capability Package** is the authoritative external handoff for one
capability (adapter or merged). Required conceptual contents:

- Inference weights / adapter files (no training sidecars)
- Root **`artifact.json`** for validation resolution and model normalize
- Optional checksums
- Optional training-local `manifest.json` (diagnostics only)

See [Lifecycle](lifecycle.md) and [Artifact Output Pipeline](artifact_output_pipeline.md).

---

## Capability lifecycle

```text
Dataset (capability corpus)
    → Capability training run
    → Checkpoint (internal)
    → Training evaluation (optional gates)
    → Capability Package (Drive publish)
    → ArtifactBundle (export inventory; sibling)
    → aiodoo-validation (certification)
    → aiodoo-model registry publish
    → Release / promotion (model)
    → Runtime consume (outside training)
```

---

## Capability vs Product

| Capability examples | Product examples |
|---------------------|------------------|
| coding, repair, execution | **Development** (composition of development skills) |
| planner, conversation, approval, evaluation | **Reasoning** (composition of reasoning skills) |
| context | May be bound into products by model policy; still trained as its own capability |

Training never emits a Development or Reasoning package.

---

## Context capability decision (Phase B0)

### Recommendation: **A — Context is an independent capability**

**Evidence inspected:**

- Dedicated generator under `aiodoo-datasets/generators/context`
- Dedicated protocol schema (`query` / `artifacts` / `graph`) distinct from planner and conversation records
- Large corpus (`context_v1_0.jsonl`, ~50k records)
- Dedicated training pack `configs/training/context/` publishing `aiodoo-context`
- `aiodoo-validation` has no `context` capability pack yet (coding/repair/execution/planner/conversation/approval/evaluation only) — this is a **validation backlog**, not proof that context is infrastructure

### Options considered

| Option | Verdict |
|--------|---------|
| **A. Independent capability** | **Accepted** |
| B. Belongs to Planner | Rejected — different task (code/graph localization vs planning) |
| C. Belongs to Conversation | Rejected — different task (retrieval/localization vs dialogue) |
| D. Internal infrastructure | Rejected — trained LoRA + protocol dataset is a skill artifact, not plumbing |

### Advantages of A

- Matches datasets and training layout already shipped
- Keeps independent train/eval/publish cadence
- Allows validation to add a context pack without merging skills
- Scales if more retrieval/localization skills appear later

### Disadvantages of A

- Product composition must decide how Development/Reasoning bind `context`
- Validation gap until a context pack exists
- One more adapter to store and promote

### Migration cost

- **Low for training** — already independent
- **Docs/terminology only** in B0
- **Validation** may add a pack later (frozen validation extends by registration)

### Implications

| Plane | Implication |
|-------|-------------|
| Training | Keep `context` in `TRAINING_IDS` and packs; treat as capability in metadata (implementation phase) |
| Validation | Future capability pack; until then, structural checks may still use shared adapter artifact types |
| Runtime | Resolve `aiodoo-context` (or registry id) when localization skill is needed |
| Products | Model policy decides binding; training does not |

### Long-term

Retain **context** as a permanent capability id. Do not fold it into planner or
conversation without a new ecosystem ADR and dataset redesign.

---

## Non-goals

- CapabilityRegistry module inside training
- Cross-capability weight chaining in training
- Treating `TrackingCapability` as a skill capability
