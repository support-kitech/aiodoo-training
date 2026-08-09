# Canonical Training Contract (TR-2)

**Living Training SoT.** System source remains authoritative.  
**System SoT:** `aiodoo-core/docs/SYSTEM.md`  
**Package:** `aiodoo_training.system_training_contract`  
**Version:** `SYSTEM_TRAINING_CONTRACT_VERSION = 1.0.0`  
**Projection:** `PROJECTION_VERSION = 1.0.0`

Training teaches the System. Training does **not** define the System.

---

## Boundary

```
Frozen AIODOO System
        ↓
Canonical Training Contract (this document + package)
        ↓
Training record schemas
        ↓
Historical Projection Layer
        ↓
Existing / future datasets
        ↓
Adapters / validation / artifacts
```

Permanent rule: System → defines behavior. Training → teaches behavior.
Runtime must never depend on Training (ECO-1).

---

## Schema version

| Field | Value |
|-------|--------|
| `training_contract_version` | `1.0.0` (System Training Contract) |
| Distinct from | `aiodoo_training.contract.TRAINING_CONTRACT_VERSION` (`aiodoo_contract` provider plane) |

Legacy / historical dataset schemas are classified as **legacy**, not current.

---

## Record types

| `record_type` | System contract | Purpose |
|---------------|-----------------|---------|
| `capability_intent` | `execution.capability_intent` | Preferred Engineering `domain.intent` + args |
| `execution_work_unit` | `execution.work_unit` | ExecutionWorkUnit WHAT surface |
| `planning_decision` | `intelligence.planning` | Goal → steps / COMPLETE |
| `observation` | `execution.observation` | Public observation envelope |
| `engineering_feedback` | `execution.engineering_feedback` | Current-cycle feedback |
| `engineering_state` | `execution.engineering_state` | Current-cycle state |
| `decision_context` | `execution.engineering_decision_context` | Model-facing bounded context |
| `loop_decision` | `intelligence_loop.decision` | REPLAN / COMPLETE / ESCALATE / … |
| `evaluation_judgment` | `capability.evaluation` | EvaluationRequest → EvaluationResponse judgment (AT-7.4) |

Common envelope fields: `training_contract_version`, `record_type`, `record_id`,
`system_contract`, `provider_capability`, `domain_specialization`, `provenance`,
`metadata`, plus type-specific `input` / `expected_output` / `evidence`.

---

## Provider vs Engineering taxonomy

| Plane | Namespace field | IDs | Role |
|-------|-----------------|-----|------|
| **Provider / adapter** | `provider_capability` | coding, repair, execution, context, planner, conversation, approval, evaluation | LoRA specialization packs |
| **Engineering WHAT** | `capability_id` / plan `action` | preferred `workspace.*`, `repository.*`, `execution.*`, `communication.*`, `diagnostics.*`, `artifact.*`, `validation.run` | Model-facing Execution |

**Never merge.** Prefer `PREFERRED_ENGINEERING_CAPABILITY_IDS` for new Training data.
Transitional System aliases (`shell`, `read`, `edit`, …) remain System-registered but
are **not** canonical Training vocabulary.

---

## Model-facing WHAT vs HOW

| Class | Examples |
|-------|----------|
| **A. Model-facing WHAT** | preferred Engineering IDs, objectives, Work Unit expected outputs, loop decision kinds |
| **B. System-generated evidence** | observation status/kind, feedback objective_state, blockers |
| **C. Historical metadata** | `provenance.source_*`, legacy schema version |
| **D. Training-only metadata** | projection_envelope, adapter pack labels, curriculum tags |
| **E. Forbidden HOW** | see below |

### Forbidden HOW vocabulary (never teach as Engineering actions)

- `local_workspace`, `local_git`, `local_program`, `local_validation`, `local_artifact`, `local_diagnostics`, …
- Strategy / Resolver / ImplementationFramework IDs
- Backend identifiers; `implementation_id` as model arg
- Shell / git / pytest / ruff / mypy as Intelligence capability vocabulary
- Hub IDs, vendor identities, `adapters_required`

---

## Historical projection

```
Historical Record
        ↓
Historical Classification (provider pack / Odoo / Protocol V1 / …)
        ↓
Projection (`project_historical_record`)
        ↓
Canonical Training Record **or** explicit non-success status
```

### Projection statuses

| Status | Meaning |
|--------|---------|
| `projected` | Semantic equivalence proven; canonical record emitted |
| `partially_projected` | Provider/domain preserved; not full Engineering equivalence |
| `unsupported` | No safe mapping (do not fabricate IDs) |
| `rejected` | Forbidden HOW / invalid shape |

Provenance on every result: `source_dataset`, `source_record_id`,
`source_schema_version`, `projection_version`, `projection_status`.

### Rules by dataset (summary)

| Dataset | Rule |
|---------|------|
| **planner** | Protocol V1 actions (`create_file`, …) → **unsupported** until explicit Engineering map |
| **coding** | Provider specialization → **partially_projected**; do not auto-convert to Work Units |
| **repair** | Project to `execution.repair` **only** when explicitly equivalent; else partial |
| **execution** | Lossy apply-artifact / missing Engineering ID → **unsupported**; need FP2-native WU |
| **context** | Odoo retrieval specialization → preserve; not Work Units |
| **conversation** | Unsupported for DecisionContext unless semantics proven |
| **evaluation** | FP2 family `evaluation_judgment` only (AT-7.4); not Continuity/observation/feedback |
| **approval** | `approve`/`reject`/`modify` → `loop_decision` when present |

---

## Odoo specialization

`domain_specialization = "odoo"` (or unset / `generic`).

Generic System Contract **+** domain specialization. Odoo corpora remain legitimate.
Do not claim Odoo examples are architecture-generic.

---

## Development / Reasoning adapter mapping

| Adapter | Provider capabilities | May contain FP2-native records |
|---------|----------------------|--------------------------------|
| **Development** | coding, repair, execution, context | Yes (independent) |
| **Reasoning** | planner, conversation, approval, evaluation | Yes (independent) |

No adapter chaining. No required adapter order. No adapter is a prerequisite for another.

---

## Validation requirements (contract for future certification)

For every canonical record:

1. Schema + `training_contract_version` validity  
2. Preferred Engineering `capability_id` where Engineering plane applies  
3. Forbidden HOW rejection  
4. Required fields per record type  
5. Provenance present when projected  
6. Deterministic `to_dict()` serialization  
7. Provider pack IDs only in `provider_capability`

**Separate from** System `validation.run` and from legacy `aiodoo-validation`
Protocol V1 certification. TR-2 defines the contract; it does not redesign certification.

---

## Generator posture (TR-2)

| Path | Action |
|------|--------|
| Existing Protocol/Odoo generators | **Preserve** (historical); classify as legacy |
| Projection functions | **Shipped** (`projection.py`) |
| New FP2-native generators | **TR-3+** — do not mass-regenerate here |
| Production JSONL | **Untouched** |

---

## Migration strategy

1. Keep historical datasets readable as legacy.  
2. Project selectively with explicit status (no silent rewrite).  
3. Generate FP2-native fixtures under `fixtures/fp2/` / `datasets/fp2/` (TR-3).  
4. Scale generation only after fixture quality gates (TR-4+).  
5. Certify adapters against preferred Engineering WHAT + provider packs.  
6. Never make Runtime import Training.  
7. TR-7 domain specialization is semantic-content driven (not provenance-only);
   unjustified labels are corrected or quarantined on versioned derivatives.

---

## Tests

- `tests/unit/test_tr2_system_training_contract.py` — contract + projection  
- `tests/unit/test_tr3_fp2_native_generators.py` — generators + fixtures  
- `tests/unit/test_tr4_fp2_quality.py` … `test_tr7_p1_fixes.py` — quality through readiness  

See also `docs/FP2_NATIVE_CORPORA.md`, `docs/FP2_PACK_EVALUATION.md`.
