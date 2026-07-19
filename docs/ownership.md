# Repository Ownership

**Status:** Authoritative ownership specification  
**Binding:** [ADR-0006](adr/0006-repository-boundaries.md), [ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md)  
**Glossary:** [terminology.md](terminology.md)

`aiodoo-training` is the **training plane**. It does not own certification,
registry, product composition, deployment, or agent runtime.

---

## Ownership matrix

| Responsibility | Training | Datasets | Validation | Model | Core / Runtime |
|----------------|:--------:|:--------:|:----------:|:-----:|:--------------:|
| Dataset generation / protocol corpora | | **Owns** | | | |
| Dataset consumption (JSONL load) | **Owns** | Supplies | | | |
| Capability identity (catalog / training ids) | **Owns** (train) | Labels data | Profiles | Metadata | Consumes |
| Capability training / LoRA | **Owns** | | | | |
| Experiment / run tracking | **Owns** | | | | |
| Checkpoint / resume | **Owns** | | | | |
| Training evaluation / quality gates | **Owns** | | | | |
| Capability Package generation (Drive publish) | **Owns** | | Consumes | Consumes | |
| ArtifactBundle export | **Owns** | | | May use roles later | |
| Package producer metadata (`artifact.json`, export_manifest) | **Owns** | | Reads | Normalizes | |
| Capability certification / oracles | | | **Owns** | Stores refs | |
| Registry / storage of artifacts | | | | **Owns** | |
| Registry publish | | | | **Owns** | |
| Product composition (Development / Reasoning) | | | | **Owns** | |
| Compatibility / promotion / release channels | | | | **Owns** | |
| Deployment handoff | | | | Resolve/load | Ops / runtime |
| Inference / serving / agent workflows | | | | Load plans | **Owns** |
| Runtime profiles | | | | | **Owns** |

---

## What training owns (detail)

- Training orchestration, curriculum/packing/sampling, distributed readiness
- Checkpoints and same-run resume
- Training-local evaluation and export
- Drive layout routing (`ArtifactOutputManager`)
- Capability Package and ArtifactBundle **production**
- Experiment summaries, metrics history, config snapshots
- Fingerprints and export integrity at produce time

## What training never owns

- Dataset authoring
- Validation oracles, certification labels, evidence trees
- Model registry mutation, promotion, compatibility policy authority
- Product packages (Development / Reasoning)
- Deployment orchestration
- Inference, vLLM/SGLang, agent runtime (`aiodoo-core`)

---

## Cross-repo contract style

Contracts are **on-disk files + protocol versions**, never shared live Python
objects across repositories. Training must not import `aiodoo_model` or
`aiodoo_validation`. Consumers must not import `aiodoo_training`.

See also: [Lifecycle](lifecycle.md), [Metadata ownership](metadata_ownership.md).
