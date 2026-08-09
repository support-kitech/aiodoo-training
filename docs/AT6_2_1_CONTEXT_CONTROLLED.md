# AT-6.2.1 — Controlled FP2 Context Corpus Expansion & Readiness Audit

**Living.** Follows AT-6.2 (`CONTEXT_GENERATOR_READY`).  
**DATA / GENERATION / QUALITY only.** Does **not** train Context.  
Does **not** modify `controlled_batch_2` or AT-6.2 fixtures.

---

## Verdict

**CONTEXT_CORPUS_READY**

Ready for a later authorized Context path-smoke / adapter-training phase.
Does **not** mean `CONTEXT_ADAPTER_READY`.

---

## AT-6.2 baseline (unchanged)

| Item | Value |
|------|------:|
| Fixtures version | `fp2-context-1.0.0` |
| Fixture count | 26 |
| Allowed families | capability_intent, observation |

---

## Controlled corpus

| Item | Value |
|------|-------|
| Version | `fp2-context-controlled-1.0.0` |
| Target | 200–300 (preferred ~250) |
| **Actual** | **261** |
| Scenario families | 60 |
| Split strategy | `fp2-split-1.0.0` via `metadata.scenario_family` |

### Locations

- `aiodoo-training/fixtures/fp2/context_controlled_1/`
- `aiodoo-datasets/datasets/fp2/context_controlled_1/`

### Contents

`context_native.jsonl`, `capability_intent.jsonl`, `observation.jsonl`,
`pack_context.jsonl`, `splits.jsonl`, `manifest.json`, `quality_report.json`,
`generation_metadata.json`

---

## Distributions

| Metric | Count |
|--------|------:|
| capability_intent | 103 |
| observation | 158 |
| Odoo | 158 |
| Generic | 103 |
| workspace.search | 141 |
| workspace.navigate | 40 |
| workspace.read | 53 |
| repository.inspect | 27 |

---

## Quality scorecard

| Metric | Result |
|--------|--------|
| Native records | 261 |
| Capability intents | 103 |
| Observations | 158 |
| Odoo | 158 |
| Generic | 103 |
| workspace.search | 141 |
| workspace.navigate | 40 |
| workspace.read | 53 |
| repository.inspect | 27 |
| Scenario families | 60 |
| Exact duplicate groups | 0 |
| Normalized duplicate groups | 0 |
| Forbidden HOW | 0 |
| Taxonomy violations | 0 |
| Negative contamination | 0 |
| Train | 198 |
| Validation | 52 |
| Test | 11 |
| Family leakage | 0 |
| Pack validity | PASS (`dataset_type=context`) |

**Scenario-family mechanism:** every intent/observation shares
`metadata.scenario_family`; `assign_split` keys on `family:<id>` so related
variants never straddle train/val/test.

**Split note:** with 60 families, hash buckets yield ~76% / 20% / 4% rather than
exact 80/10/10. Family isolation is preserved; absolute test N is small but clean.

---

## Negatives / safety

AT-6.2 Context negatives still match. Corpus excludes `quality_corpus=negative*`.
No legacy `context_v1_0` contamination.

---

## Legacy Context decision

Native controlled corpus is **sufficient to justify a later Context path smoke**
without requiring legacy projection first.

A future **`Context Legacy → FP2 Projection`** phase remains **optional /
separate** for scale and Odoo retrieval breadth — **not** started here.

---

## controlled_batch_2 checksum

| | Value |
|--|-------|
| Before | `728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227` |
| After | `728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227` |

AT-6.2 fixtures (`fp2-context-1.0.0`, 26 records) **not overwritten**.

---

## Files

**Added**

- `aiodoo_training/system_training_contract/generators/context_controlled.py`
- `tests/unit/test_at621_context_controlled.py`
- `fixtures/fp2/context_controlled_1/*`
- `aiodoo-datasets/datasets/fp2/context_controlled_1/*`
- `docs/AT6_2_1_CONTEXT_CONTROLLED.md` (this file)

**Changed**

- `generators/__init__.py` (exports)
- `docs/STATUS.md`
- `docs/TRAINING_CONTRACT_TARGET.md`

---

## Tests

- `test_at621_context_controlled.py`
- Re-run AT-6.2 + TR-2…TR-7 as appropriate

---

## Remaining gaps

- Test split absolute size small (family-hash skew)
- search still plurality of caps (expected for retrieval; others ≥10)
- Not a production train/val/test adapter pack under skill configs
- Legacy 50k still unprojected
- No Context LoRA / certification

---

## Next recommendation

**STOP.**

When authorized: **Context path smoke** (AT-6-style, analogous to AT-5.1) using
`fp2-context-controlled-1.0.0` — still no certification, no adapter merge,
no automatic legacy projection.
