# Status — aiodoo-training

**Living document.** Code on `main` is the only implementation Source of Truth
for this **Training** repository.  
**System documentation SoT (separate ecosystem):** `aiodoo-core/docs/SYSTEM.md`  
**Permanent branch:** `main`  
**Historical evidence:** `docs/archive/`

## Purpose

Train optional capability adapters; publish Capability Packages. **Training
never defines the System.** The System must operate correctly with foundation
models only; adapters are optional.

## Current implementation (on main)

| Item | Status |
|------|--------|
| Capability training plane | Shipped / frozen surface on `main` |
| Emits Development / Reasoning product packages | **Never** (ownership: `aiodoo-model`) |
| Product composition | Out of scope |
| Required for System correctness | **No** |
| ECO-1 boundary | **Audited** — Running System does not import this package; adapters remain optional artifacts |
| Dataset/System contract alignment | **TR-7 COMPLETE** — pack readiness: **READY_FOR_TRAINING** on `controlled_batch_2`. See `docs/FP2_PACK_EVALUATION.md`. |
| Adapter training pipeline audit | **AT-1 COMPLETE** — **READY_WITH_REQUIRED_FIXES**. See `docs/AT1_PIPELINE_AUDIT.md`. |
| FP2 train-path fixes + smoke | **AT-2.1 COMPLETE** — **SMOKE_PASS**. See `docs/AT2_FP2_SMOKE.md`. |
| First controlled skill adapter | **AT-3 COMPLETE** — **AT3_TRAINING_PATH_PASS** (coding). See `docs/AT3_CODING.md`. |
| Coding coverage audit | **AT-3.1 COMPLETE** — **CODING_COVERAGE_CONFIRMED**. See `docs/AT3_1_CODING_COVERAGE_AUDIT.md`. |
| Controlled repair skill adapter | **AT-4 COMPLETE** — **AT4_TRAINING_PATH_PASS**. See `docs/AT4_REPAIR.md`. |
| Execution path smoke | **AT-5.1 COMPLETE** — **AT5_1_TRAINING_PATH_PASS** (4-step CPU smoke; full 522-train not run). See `docs/AT5_1_EXECUTION_SMOKE.md`. |
| Context coverage audit | **AT-6.1 COMPLETE** — **CONTEXT_COVERAGE_GAP** (0 FP2 Context labels). See `docs/AT6_1_CONTEXT_COVERAGE_AUDIT.md`. |
| Context generator / mapping | **AT-6.2 COMPLETE** — **CONTEXT_GENERATOR_READY** (`fp2-context-1.0.0` fixtures). See `docs/AT6_2_CONTEXT_GENERATOR.md`. Does **not** authorize Context training. |
| Context controlled corpus | **AT-6.2.1 COMPLETE** — **CONTEXT_CORPUS_READY** (`fp2-context-controlled-1.0.0`, 261 records). See `docs/AT6_2_1_CONTEXT_CONTROLLED.md`. Does **not** authorize Context training. |
| Context path smoke | **AT-6.3 COMPLETE** — **AT6_3_TRAINING_PATH_PASS** (16/4 smoke, 4 steps). See `docs/AT6_3_CONTEXT_SMOKE.md`. Does **not** certify. |
| Reasoning coverage audit | **AT-7.1 COMPLETE** — **REASONING_COVERAGE_MIXED** (planner confirmed; conversation/approval/evaluation data-phase required). See `docs/AT7_1_REASONING_COVERAGE_AUDIT.md`. |
| Planner path smoke | **AT-7.2 COMPLETE** — **AT7_2_TRAINING_PATH_PASS** (16/4 smoke, 4 steps). See `docs/AT7_2_PLANNER_SMOKE.md`. Does **not** certify. |
| Reasoning sparse data | **AT-7.3 COMPLETE** — **REASONING_SPARSE_DATA_PARTIAL** (Conversation + Approval corpora ready; Evaluation semantics unresolved at time of phase). See `docs/AT7_3_REASONING_SPARSE_DATA.md`. Does **not** train. |
| Evaluation contract/mapping | **AT-7.4 COMPLETE** — **EVALUATION_MAPPING_READY** (`evaluation_judgment`). See `docs/AT7_4_EVALUATION_CONTRACT.md`. |
| Evaluation controlled corpus | **AT-7.5 COMPLETE** — **EVALUATION_CORPUS_READY** (`fp2-evaluation-controlled-1.0.0`, 252). See `docs/AT7_5_EVALUATION_CONTROLLED.md`. Does **not** train. |
| Evaluation path smoke | **AT-7.6 COMPLETE** — **AT7_6_TRAINING_PATH_PASS** (16/4 smoke, 4 steps). See `docs/AT7_6_EVALUATION_SMOKE.md`. Does **not** certify. |
| Conversation path smoke | **AT-7.7 COMPLETE** — **AT7_7_TRAINING_PATH_PASS** (16/4 smoke, 4 steps). See `docs/AT7_7_CONVERSATION_SMOKE.md`. Does **not** certify. |
| Approval path smoke | **AT-7.8 COMPLETE** — **AT7_8_TRAINING_PATH_PASS** (16/4 smoke, 4 steps). See `docs/AT7_8_APPROVAL_SMOKE.md`. Does **not** certify. |
| Training / System boundary | **Audited** — adapter handoff path proven; foundation-only preserved. Training **FROZEN FOR NOW** (no production adapters). See `docs/TRAINING_SYSTEM_BOUNDARY.md`. |
| Training Contract Target | **Living** — `docs/TRAINING_CONTRACT_TARGET.md` |
| Canonical Training System Contract | **Shipped** — `docs/TRAINING_SYSTEM_CONTRACT.md` |
| FP2-native corpora | **Shipped** — fixtures + `controlled_batch_1` + TR-7 `controlled_batch_2` |
| FP2 corpus quality | **Shipped** — `docs/FP2_CORPUS_QUALITY.md` |
| FP2 controlled batch | **Shipped** — `docs/FP2_CONTROLLED_BATCH.md` |
| FP2 pack evaluation | **Shipped** — `docs/FP2_PACK_EVALUATION.md` |

## Living docs

- `docs/TRAINING_SYSTEM_BOUNDARY.md` — Training/System boundary audit + freeze
- `docs/AT7_8_APPROVAL_SMOKE.md` — AT-7.8 Approval path smoke
- `docs/AT7_7_CONVERSATION_SMOKE.md` — AT-7.7 Conversation path smoke
- `docs/AT7_6_EVALUATION_SMOKE.md` — AT-7.6 Evaluation path smoke
- `docs/AT7_5_EVALUATION_CONTROLLED.md` — AT-7.5 controlled Evaluation corpus
- `docs/AT7_4_EVALUATION_CONTRACT.md` — AT-7.4 Evaluation FP2 contract/mapping decision
- `docs/AT7_3_REASONING_SPARSE_DATA.md` — AT-7.3 Reasoning sparse-skill data readiness
- `docs/AT7_2_PLANNER_SMOKE.md` — AT-7.2 Planner path smoke
- `docs/AT7_1_REASONING_COVERAGE_AUDIT.md` — AT-7.1 Reasoning coverage audit
- `docs/AT6_3_CONTEXT_SMOKE.md` — AT-6.3 Context path smoke
- `docs/AT6_2_1_CONTEXT_CONTROLLED.md` — AT-6.2.1 controlled Context corpus
- `docs/AT6_2_CONTEXT_GENERATOR.md` — AT-6.2 Context generator & mapping
- `docs/AT6_1_CONTEXT_COVERAGE_AUDIT.md` — AT-6.1 Context coverage audit
- `docs/AT5_1_EXECUTION_SMOKE.md` — AT-5.1 Execution path smoke
- `docs/AT4_REPAIR.md` — AT-4 controlled repair skill adapter
- `docs/AT3_1_CODING_COVERAGE_AUDIT.md` — AT-3.1 coding selection audit
- `docs/AT3_CODING.md` — AT-3 controlled coding skill adapter
- `docs/AT2_FP2_SMOKE.md` — AT-2 / AT-2.1 FP2 loader/export/provenance + local smoke
- `docs/AT1_PIPELINE_AUDIT.md` — AT-1 training pipeline readiness
- `docs/FP2_PACK_EVALUATION.md` — TR-6/TR-7 training-pack readiness
- `docs/FP2_CONTROLLED_BATCH.md` — TR-5/TR-7 controlled batches
- `docs/FP2_CORPUS_QUALITY.md` — TR-4 quality gates
- `docs/FP2_NATIVE_CORPORA.md` — TR-3 generators + fixture inventory
- `docs/TRAINING_SYSTEM_CONTRACT.md` — TR-2 Canonical Training Contract
- `docs/TRAINING_CONTRACT_TARGET.md` — alignment target + gap matrix
- `docs/architecture.md`, `docs/capability_model.md`, `docs/product_model.md`, `docs/lifecycle.md`, `docs/ownership.md`
- `docs/CONTRACT_ADOPTION.md`, `docs/adr/`, `PRODUCTION_TRAINING.md`
- `README.md`, `CHANGELOG.md`
