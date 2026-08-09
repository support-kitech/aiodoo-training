# Training Contract Target (TR-1 … TR-6)

**Living Training SoT for alignment.** System source remains authoritative.  
**System SoT:** `aiodoo-core/docs/SYSTEM.md`  
**Canonical contract (TR-2):** `docs/TRAINING_SYSTEM_CONTRACT.md`  
**Package:** `aiodoo_training.system_training_contract` (`1.0.0`)

Training teaches the System. Training does **not** define the System.

---

## Frozen System contracts Training must teach

### Model-facing WHAT (train)

| Contract | System source (examples) | Canonical record |
|----------|---------------------------|------------------|
| Engineering Capability intents | `execution/capabilities/ids.py` | `capability_intent` |
| Plan / Work Unit actions | `intelligence/planning.py`, `execution/work_unit/types.py` | `planning_decision`, `execution_work_unit` |
| Observation envelopes | `execution/observations/` | `observation` |
| EngineeringFeedback | `execution/feedback/types.py` | `engineering_feedback` |
| EngineeringState | `execution/engineering_state/types.py` | `engineering_state` |
| EngineeringDecisionContext | `execution/engineering_state/decision_context.py` | `decision_context` |
| Loop decisions | `intelligence_loop/` — **REPLAN / COMPLETE / ESCALATE** | `loop_decision` |
| Foundation **roles** | Development / Reasoning | `provider_capability` plane |

### System-owned HOW (never emit as model vocabulary)

| Contract | Notes |
|----------|--------|
| `local_*` implementations | `local_workspace`, `local_git`, `local_program`, `local_validation`, … |
| Strategy / Resolver / ImplementationFramework | Backend selection |
| Shell / git / pytest / ruff as Intelligence actions | HOW only |
| Hub IDs / vendors | Configuration |
| `adapters_required=true` | Forbidden — foundation-only permanent |

### Provider vs Engineering taxonomy

| Plane | IDs | Role |
|-------|-----|------|
| **Provider / adapter packs** | `coding`, `repair`, `execution`, `context`, `planner`, `conversation`, `approval`, `evaluation` | Development vs Reasoning adapters |
| **Engineering Execution** | preferred `workspace.*`, `repository.*`, … | Plan / Work Unit actions |

Do **not** conflate these planes.

---

## Development vs Reasoning adapters (unchanged architecture)

| Adapter role | Capabilities |
|--------------|--------------|
| **Development** | coding, repair, execution, context |
| **Reasoning** | planner, conversation, approval, evaluation |

Adapters remain **independent** reusable artifacts (no chain).

---

## Status through TR-7

| Item | Status |
|------|--------|
| Canonical Training System Contract | **Shipped** |
| FP2-native generators + fixtures | **TR-3 COMPLETE** |
| Corpus quality / readiness | **TR-4 COMPLETE** |
| Controlled batch 1 | **TR-5 COMPLETE / PASS** (immutable) |
| Pack evaluation | **TR-6 COMPLETE** — **READY_WITH_REQUIRED_DATA_FIXES** (batch_1) |
| P1 data fixes + re-evaluation | **TR-7 COMPLETE** — **READY_FOR_TRAINING** (`controlled_batch_2`) |
| Adapter training pipeline audit | **AT-1 COMPLETE** — **READY_WITH_REQUIRED_FIXES** (`docs/AT1_PIPELINE_AUDIT.md`) |
| FP2 path fixes + smoke | **AT-2.1 COMPLETE** — **SMOKE_PASS** (`docs/AT2_FP2_SMOKE.md`) |
| First controlled skill adapter | **AT-3 COMPLETE** — **AT3_TRAINING_PATH_PASS** coding (`docs/AT3_CODING.md`) |
| Coding coverage audit | **AT-3.1 COMPLETE** — **CODING_COVERAGE_CONFIRMED** |
| Controlled repair skill adapter | **AT-4 COMPLETE** — **AT4_TRAINING_PATH_PASS** (`docs/AT4_REPAIR.md`) |
| Execution path smoke | **AT-5.1 COMPLETE** — **AT5_1_TRAINING_PATH_PASS** (`docs/AT5_1_EXECUTION_SMOKE.md`); full Execution train not run |
| Context coverage audit | **AT-6.1 COMPLETE** — **CONTEXT_COVERAGE_GAP** (`docs/AT6_1_CONTEXT_COVERAGE_AUDIT.md`) |
| Context generator / mapping | **AT-6.2 COMPLETE** — **CONTEXT_GENERATOR_READY** (`docs/AT6_2_CONTEXT_GENERATOR.md`); fixtures only — **do not** start AT-6 train |
| Context controlled corpus | **AT-6.2.1 COMPLETE** — **CONTEXT_CORPUS_READY** (`docs/AT6_2_1_CONTEXT_CONTROLLED.md`); path smoke / adapter train not authorized |
| Context path smoke | **AT-6.3 COMPLETE** — **AT6_3_TRAINING_PATH_PASS** (`docs/AT6_3_CONTEXT_SMOKE.md`); full train / certify not authorized |
| Reasoning coverage audit | **AT-7.1 COMPLETE** — **REASONING_COVERAGE_MIXED** (`docs/AT7_1_REASONING_COVERAGE_AUDIT.md`); do not train sparse Reasoning skills yet |
| Planner path smoke | **AT-7.2 COMPLETE** — **AT7_2_TRAINING_PATH_PASS** (`docs/AT7_2_PLANNER_SMOKE.md`); full Planner / certify not authorized |
| Reasoning sparse data | **AT-7.3 COMPLETE** — **REASONING_SPARSE_DATA_PARTIAL** (`docs/AT7_3_REASONING_SPARSE_DATA.md`); Conversation/Approval corpora ready |
| Evaluation contract/mapping | **AT-7.4 COMPLETE** — **EVALUATION_MAPPING_READY** (`docs/AT7_4_EVALUATION_CONTRACT.md`) |
| Evaluation controlled corpus | **AT-7.5 COMPLETE** — **EVALUATION_CORPUS_READY** (`docs/AT7_5_EVALUATION_CONTROLLED.md`) |
| Evaluation path smoke | **AT-7.6 COMPLETE** — **AT7_6_TRAINING_PATH_PASS** (`docs/AT7_6_EVALUATION_SMOKE.md`); full train / certify not authorized |
| Conversation path smoke | **AT-7.7 COMPLETE** — **AT7_7_TRAINING_PATH_PASS** (`docs/AT7_7_CONVERSATION_SMOKE.md`); full train / certify not authorized |
| Approval path smoke | **AT-7.8 COMPLETE** — **AT7_8_TRAINING_PATH_PASS** (`docs/AT7_8_APPROVAL_SMOKE.md`); full train / certify not authorized |
| Training / System boundary | **FROZEN FOR NOW** — path smokes + packaging sufficient; no production adapters. See `docs/TRAINING_SYSTEM_BOUNDARY.md` |
| Adapter training / LoRA / Colab | **path smokes complete**; production trains / certify deferred until System needs them |

---

## Gap severity (post TR-7)

| ID | Gap | Severity |
|----|-----|----------|
| G1–G2, G5, G9–G11 | Contract, fixtures, quality, coverage | **Closed** |
| G14 | Continuity family volume | **Closed** (77/77/77 on batch_2) |
| G15 | Ambiguous Odoo/generic labels | **Closed** (0 ambiguous) |
| G3–G4 | Legacy Protocol V1 planner/execution | **P1** (historical; out of FP2 path) |
| G6 | Colab/Hub legacy | **P2** |
| G16 | Train-only caps / overrepresented caps | **P2** |
| G7–G8 | Context eval; product composition | **P3** |
| G13 | Adapter training | **AT-4 repair path PASS** (after AT-3 coding); other skills + certification pending |

---

## Proposed next (do not start automatically)

AT-7.8: **AT7_8_TRAINING_PATH_PASS**. Training/System boundary audited.
**TRAINING: FROZEN FOR NOW.** Do not start production adapter trains, AT-7.9,
or AT-8 unless a System contract change requires Training work. See
`docs/TRAINING_SYSTEM_BOUNDARY.md`. Primary focus: AIODOO System vertical slice.

---

## Permanent rules

1. System source wins over Training docs.  
2. Training produces artifacts; System never depends on Training (ECO-1).  
3. Foundation-only / zero adapters remains permanent.  
4. Odoo specialization is allowed; Odoo ≠ AIODOO System architecture.  
5. Do not fabricate Engineering IDs when projection cannot prove equivalence.  
6. No commit/push from TR implementation chats without separate authorization.
