# Product Model

**Status:** Authoritative product specification (training perspective)  
**Binding:** [ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md)  
**Related:** [Capability Model](capability_model.md), [Ownership](ownership.md), Ecosystem ADR-0001 — AIODOO Model Lifecycle

---

## Definition

A **Product** is a **composed** consumer-facing model offering. Products are
**not** trained in `aiodoo-training`.

Canonical products:

| Product | Typical capability composition (policy in `aiodoo-model`) |
|---------|----------------------------------------------------------|
| **Development** | coding, repair, execution (+ optional bindings such as context per model policy) |
| **Reasoning** | planner, conversation, approval, evaluation (+ optional bindings per model policy) |

Exact binding rules, release channels, and aliases live in **`aiodoo-model`**.
This document only fixes **ownership** from the training plane.

---

## Ownership

| Concern | Owner |
|---------|-------|
| Train each capability adapter | `aiodoo-training` |
| Compose Development / Reasoning | `aiodoo-model` |
| Publish / promote product releases | `aiodoo-model` |
| Validate each capability | `aiodoo-validation` |
| Invoke products at runtime | `aiodoo-core` / runtime |

```text
Training creates Capabilities.
Model composes Products.
```

---

## What training must not do

- Emit a package named or typed as Development or Reasoning
- Chain capability adapters as a substitute for product composition
  (`resume_from` across capabilities is forbidden; same-run resume only)
- Treat Drive names like `aiodoo-coding` as product names

`aiodoo-<capability>` is a **Capability Package directory name**, not a Product.

---

## Relationship diagram

```text
Base model
   ├── capability: coding      ──┐
   ├── capability: repair      ──┼──→ Product: Development   (aiodoo-model)
   ├── capability: execution   ──┘
   ├── capability: context     ──→ (binding policy in aiodoo-model)
   ├── capability: planner     ──┐
   ├── capability: conversation──┤
   ├── capability: approval    ──┼──→ Product: Reasoning     (aiodoo-model)
   └── capability: evaluation  ──┘
```

---

## Documentation note

Older config comments saying “product adapter” mean **capability adapter**.
Use [terminology.md](terminology.md) going forward.
