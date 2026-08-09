# AT-6.1 — Context Corpus Coverage Audit (READ-ONLY)

**Living.** Pre-AT-6 gate.  
**Does not** train Context. **Does not** modify `controlled_batch_2`.  
**Does not** project legacy Context into FP2.

---

## Verdict

**CONTEXT_COVERAGE_GAP**

`controlled_batch_2` (`fp2-controlled-2.0.0-tr7`) contains **zero** records with
`provider_capability=context` and **zero** with `dataset_type=context`.

AT-6 Context adapter training **must not start** until an authorized
FP2 Context generator / mapping / data phase produces a labeled population.

---

## Counts

| Selection | Count |
|-----------|------:|
| Native `provider_capability=context` | **0** |
| Development-pack `provider_capability=context` | **0** |
| Reasoning-pack `provider_capability=context` | **0** |
| `dataset_type=context` (any pack) | **0** |
| Provider ≡ dataset_type for context | **trivial** (both empty) |
| Train / val / test for context | **0 / 0 / 0** |
| Excluded from Development pack as context | **n/a** (none exist) |

Native provider distribution (for contrast): execution 641, planner 657,
repair 42, coding 26, approval 9, conversation 8, evaluation 3, **context 0**.

---

## Record-type / Engineering / domain for Context

All **empty** (no Context-labeled population).

---

## Why Context = 0 (source-level)

Composite of **B + C + E** (not an AT-3-style selection filter bug):

| ID | Finding | Evidence |
|----|---------|----------|
| **C** | Provider mapping omission | `generators/mapping.py` `_RECORD_TO_PROVIDERS` lists **no** record type that includes `"context"` |
| **B / E** | Missing Context generator coverage / dataset generation omission | No FP2 generator module emits `provider_capability="context"` (no matches under `system_training_contract/generators/`) |
| Taxonomy vs data | Contract/taxonomy **name** Context | `DEVELOPMENT_PROVIDER_CAPABILITIES` includes `context`; `TRAINING_SYSTEM_CONTRACT.md` lists Context as Development — **not instantiated in TR-7 data** |
| Prior signal | Already known P3 | TR-6/TR-7 pack eval: `context_provider_pack_underrepresented_in_development_formatter_outputs` |

This is a **coverage defect relative to the declared Development product plane**,
not an intentional “Context forbidden” policy in taxonomy. Generators simply
never assign the label; mapping never allows it on shared families.

**Not A alone:** taxonomy expects Context; absence is incomplete FP2 coverage.

**Not D alone:** Development formatter includes shared record types; it would
format Context instances if they existed with a Development-allowed record type.
The omission is upstream (no labeled natives + mapping gap).

---

## Provider mapping logic (source)

```text
taxonomy: context ∈ DEVELOPMENT_PROVIDER_CAPABILITIES
mapping:  context ∉ ∪_record_type _RECORD_TO_PROVIDERS[record_type]
generators: never set provider_capability="context"
formatter: dataset_type := provider_capability when set
→ packs never contain Context TrainingExamples
```

---

## Candidates (report only — do not relabel)

No records with literal `provider_capability=context`.

Heuristic phrase scan for “retrieve context” / “context pack” / “odoo retrieval”
found **0** natives.

Strings containing the English word “context” (e.g. observation summary
“Artifact attached to engineering context”) are **Execution** provider records —
**not** Context candidates for relabeling under current taxonomy rules.

Do **not** convert Execution/Planner/`decision_context` record **types** into
the Context **provider** (different planes).

---

## Legacy Context (separate from FP2)

| Item | Value |
|------|-------|
| File | `aiodoo-datasets/datasets/context_v1_0.jsonl` |
| Count | **50161** |
| Schema | retrieval graph: `id`, `query`, `artifacts`, `graph`, `metadata` |
| FP2 fields | **none** (`provider_capability`, `record_type`, `messages`, … absent) |
| Production config | `configs/training/context/dataset.yaml` → `context_v1_0.jsonl` (legacy Colab path) |

Contract note: Context ≈ “Odoo retrieval specialization” — legacy file matches
that *product idea*, but it is **not** FP2-native and must **not** be mixed into
AT-6 without an authorized projection/generator phase.

---

## Quality checks (corpus)

| Check | Result |
|-------|--------|
| Negatives in natives | 0 |
| Legacy flags on FP2 natives | 0 |
| Forbidden HOW hits | 0 |
| `training_contract_version` | all `1.0.0` |
| Checksum before | `728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227` |
| Checksum after | **unchanged** |

---

## Recommendation

1. **STOP** — do not start AT-6 Context training.  
2. Authorize a **data/generator phase** to:  
   - define which FP2 record families Context may label (`mapping.py`)  
   - emit native fixtures with `provider_capability=context`  
   - optionally plan legacy `context_v1_0` → FP2 projection (separate, explicit)  
3. Only after a non-zero Context population exists and is audited (like AT-3.1),
   authorize a Context path smoke analogous to AT-5.1.

---

## Files changed

Audit documentation only:

- `docs/AT6_1_CONTEXT_COVERAGE_AUDIT.md`
- status pointers in `docs/STATUS.md` / `docs/TRAINING_CONTRACT_TARGET.md` (if updated)

No corpus, config, or training code changes required for the audit itself.
