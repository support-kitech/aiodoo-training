# AT-6.2 — FP2 Context Generator & Mapping Data Phase

**Living.** Follows AT-6.1 (`CONTEXT_COVERAGE_GAP`).  
**DATA / GENERATOR / MAPPING only.** Does **not** train Context.  
Does **not** modify `controlled_batch_2`. Does **not** project legacy Context into FP2 packs.

---

## Verdict

**CONTEXT_GENERATOR_READY**

This means Context provider semantics, mapping, generator, fixtures, and quality
gates are proven. It does **not** mean `CONTEXT_ADAPTER_READY` and does **not**
authorize LoRA / AT-6 training.

---

## 1. System source evidence

| Source | Finding |
|--------|---------|
| `aiodoo-contract/.../foundations.py` `DEVELOPMENT_CAPABILITIES` | Includes `"context"` as a Development foundation capability |
| `aiodoo-training/.../taxonomy.py` `DEVELOPMENT_PROVIDER_CAPABILITIES` | Mirrors System: coding, repair, execution, **context** |
| `docs/TRAINING_SYSTEM_CONTRACT.md` | Context = “Odoo retrieval specialization → preserve; **not Work Units**” |
| `docs/capability_model.md` | Context is an **independent** capability (retrieval/localization), not Planner/Conversation |
| `aiodoo-core` Context Builder / `EngineeringDecisionContext` | **Different** plane: decision continuity / planning context assembly — **not** provider=`context` |
| Pre-AT-6.2 `mapping.py` | `context` absent from `_RECORD_TO_PROVIDERS` (AT-6.1 gap) |
| Pre-AT-6.2 generators | No emitter of `provider_capability=context` |

**Doc vs source:** Training docs and contract already named Context as Development
retrieval; System taxonomy already lists it. The defect was Training data/mapping
omission, not a missing System capability name. No System taxonomy change was made.

---

## 2. Definition — Context provider specialization

`provider_capability = context` teaches the Development **retrieval / locate**
skill:

- Find models, fields, methods, views, manifests, controllers in a workspace/repo
- Rank / report locate results as observations
- Use preferred Engineering WHAT: `workspace.search`, `workspace.navigate`,
  `workspace.read`, `repository.inspect`

It is **not**:

| Homonym | Meaning |
|---------|---------|
| `record_type=decision_context` | EngineeringDecisionContext / Decision Continuity |
| Context Builder subsystem | Runtime assembly of planning/decision evidence |
| `domain_specialization=odoo` | Domain tag only (set only when content proves Odoo) |
| Execution / Repair / Coding | Adjacent Development providers with different jobs |

---

## 3. Provider vs Engineering taxonomy

| Plane | Role for Context |
|-------|------------------|
| Provider | `context` (adapter pack id) |
| Engineering WHAT | locate/search/read/inspect IDs above |
| Forbidden HOW | Never (`local_*`, backends, shell/git as capabilities) |

`assert_no_adapter_chain()` now also asserts Context allow/reject families.

---

## 4–6. Allowed / rejected record families

| Record Type | Context Allowed? | Reason / System Evidence |
|-------------|------------------|--------------------------|
| `capability_intent` | **Yes** | Model-facing locate intents map cleanly to preferred Engineering WHAT |
| `observation` | **Yes** | `search_result` / navigate / artifact observations are locate outcomes |
| `execution_work_unit` | **No** | TRAINING_SYSTEM_CONTRACT: Context → preserve; **not Work Units** |
| `decision_context` | **No** | Decision Continuity ≠ retrieval provider (homonym trap) |
| `planning_decision` | **No** | Reasoning / Planner specialization |
| `loop_decision` | **No** | Reasoning loop control |
| `engineering_state` | **No** | Continuity state, not retrieval |
| `engineering_feedback` | **No** | Cycle feedback to Planner/Execution, not Context pack |

---

## 7. Mapping changes

File: `aiodoo_training/system_training_contract/generators/mapping.py`

- Added `"context"` to `capability_intent` and `observation` provider sets
- Added `CONTEXT_ALLOWED_RECORD_TYPES` / `CONTEXT_REJECTED_RECORD_TYPES`
- Extended `assert_no_adapter_chain()` to enforce Context family rules

---

## 8. Generator changes

File: `aiodoo_training/system_training_contract/generators/context.py`

- Independent Context generator (not folded into TR-3 `GENERATOR_NAMES`)
- Emits `fp2-context-1.0.0` fixtures only
- `emit_context_fixtures()` writes training + datasets Context trees — never `controlled_batch_2`

---

## 9–12. Fixture inventory (`fp2-context-1.0.0`)

| Metric | Value |
|--------|------:|
| Total fixtures | **26** |
| `capability_intent` | **16** |
| `observation` | **10** |
| Odoo specialization | **19** |
| Generic (unset) | **7** |

Engineering capability distribution (intent + observation capability_id):

| Capability | Count |
|------------|------:|
| `workspace.search` | 16 |
| `repository.inspect` | 4 |
| `workspace.navigate` | 3 |
| `workspace.read` | 3 |

Locations:

- `aiodoo-training/fixtures/fp2/context/`
- `aiodoo-datasets/datasets/fp2/context/`

---

## 13. Negative cases (quality-only)

Added to `quality/negatives.py`:

| Case | Expected |
|------|----------|
| `neg_context_forbidden_how` | rejected (Context + `local_workspace`) |
| `neg_decision_context_as_context_provider_schema_ok` | schema accepted; **mapping policy** rejects family (AT-6.2 tests) |
| `pos_context_locate_intent` | accepted control positive |

Negatives never enter training packs.

---

## 14. Quality gate results

| Gate | Result |
|------|--------|
| Schema / validate_record_mapping | PASS on all 26 |
| Forbidden HOW | PASS (0) |
| Provider/Engineering taxonomy | PASS |
| Deterministic serialization | PASS |
| Duplicates | PASS (unique record_ids) |
| Odoo/generic semantics | PASS (both present; Odoo only when content proves) |
| Legacy Protocol V1 contamination | PASS (none) |
| Development pack format | PASS (`dataset_type=context` for all 26) |
| TR-2 … TR-7 + AT-6.2 unit tests | **66 passed** |

---

## 15–16. Legacy Context analysis

| Item | Value |
|------|-------|
| File | `aiodoo-datasets/datasets/context_v1_0.jsonl` |
| Count | 50161 |
| Schema | `id`, `query`, `artifacts`, `graph`, `metadata` (Protocol-style retrieval graph) |
| FP2 fields | **absent** |
| Role in AT-6.2 | Historical evidence only — **not converted**, **not in fixtures** |

**LEGACY CONTEXT:**

- **Usable as historical evidence:** yes (query types, ranked artifacts, Odoo module hints)
- **Safe projection candidates (future):** query NL → capability_intent objective; top artifact → observation `ranked_artifacts`; `metadata.odoo_version` / module → domain cues
- **Unsafe/unsupported:** graph node IDs / protocol hashes as Engineering IDs; inventing Work Units; blind `provider_capability=context` without family mapping
- **Future projection phase:** **recommended** as a separate authorized phase (`Context Legacy → FP2 Projection`), with projected / partially_projected / unsupported / rejected — **not** mixed into native fixtures

---

## 17. New files

- `aiodoo_training/system_training_contract/generators/context.py`
- `tests/unit/test_at62_context_generator.py`
- `fixtures/fp2/context/*` (`context_native.jsonl`, family jsonl, `manifest.json`)
- `aiodoo-datasets/datasets/fp2/context/*` (mirror)
- `docs/AT6_2_CONTEXT_GENERATOR.md` (this file)

---

## 18. Existing files changed

- `aiodoo_training/system_training_contract/generators/mapping.py`
- `aiodoo_training/system_training_contract/generators/__init__.py`
- `aiodoo_training/system_training_contract/quality/negatives.py`
- `docs/STATUS.md`
- `docs/TRAINING_CONTRACT_TARGET.md`

---

## 19. `controlled_batch_2` checksum

| | Value |
|--|-------|
| Before | `728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227` |
| After | `728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227` |
| Modified | **No** |

---

## 20. Tests

- `tests/unit/test_at62_context_generator.py` (mapping, generator, validation, taxonomy, HOW, determinism, Odoo/generic, negatives, legacy separation, batch_2 immutability)
- Existing TR-2 … TR-7 suites re-run: PASS

---

## 21. Remaining gaps

- Fixture-scale only (26) — not a production Context adapter pack / train/val/test
- Legacy 50k not projected
- `controlled_batch_2` still has 0 Context labels (intentional; immutable)
- No Context LoRA / path smoke yet

---

## 22. Exact next recommended phase

**STOP here.** Wait for explicit authorization.

Recommended next (when authorized), in order:

1. **AT-6.2.1 / Context scale** (optional): expand native Context fixtures + quality re-audit  
2. **or Context Legacy → FP2 Projection** (separate; never silent mix)  
3. **then** Context path smoke / adapter train (only after non-zero audited population + authorization)

Do **not** start AT-6 training automatically.
