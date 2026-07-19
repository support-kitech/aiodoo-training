# Capability Package fixture strategy

**Status:** Authoritative test-infrastructure policy (not a production contract)  
**Protocol:** `protocol/v1`

## Verdict

Use **representative committed goldens** + **exhaustive parametrized generation in tests**.

Do **not** commit one static JSON file per capability × package kind.

## Why representative beats exhaustive static files

| Approach | Protection | Cost |
|----------|------------|------|
| 8 adapter JSON files differing only by `capability_id` | Low — duplicates the same protocol shape | High drift (`training_version`, churn) |
| Parametrized `build_*` over `TRAINING_IDS` | High — proves every capability identity | Low |
| Representative goldens (base + coding + repair + merged) | High — freezes protocol shape + identity-collapse sentinel | Low |

Committed goldens answer: “What does a blessed protocol/v1 package look like?”  
Parametrized tests answer: “Does every catalog capability still emit that protocol correctly?”

Those are different jobs. Exhaustive static files conflate them and rot.

## Catalog (committed)

```text
tests/fixtures/capability_packages/
  README.md                 # this file
  regenerate.py             # rebuild goldens from builders
  protocol/
    v1/
      base_models/
        qwen3-8b/artifact.json     # only production base in packs today
      adapters/
        coding/artifact.json       # primary Development skill example
        repair/artifact.json       # non-coding identity sentinel
      merged/
        coding/artifact.json       # single merged kind example
```

### Why these four (not eight × three)

1. **`base_models/qwen3-8b`** — Sole base used by production training packs. DeepSeek appears only under `future_models` and still as `family: qwen`; do not invent a DeepSeek golden until a pack trains on it.
2. **`adapters/coding`** — Primary protocol example (matches validation/model fixture culture).
3. **`adapters/repair`** — Proves `capability_id` / `adapter_type` are not hard-coded to coding while `artifact_type` stays `coding_adapter`.
4. **`merged/coding`** — Merged packages differ from adapters mainly by `artifact_type=merged_model` and omitted PEFT fields; one example is enough. Per-capability merged JSON would be pure duplication.

### Explicitly not committed

- Adapters for execution/planner/context/conversation/approval/evaluation (covered by parametrized tests)
- Merged per capability
- DeepSeek / other future bases
- Full weight trees (metadata-only goldens; layout covered by publish unit/contract tests)

## Protocol versioning

Goldens live under `protocol/v1/` so a future `protocol/v2/` can coexist without rewriting history.

Bump the directory when Capability Package **producer** JSON shape breaks (new required fields or renamed protocol kinds). Do not bump for additive optional fields.

## Generation policy

- **Source of truth for field values:** `aiodoo_training.artifacts.publish_contract` builders.
- **Committed goldens:** regenerated via `python3 tests/fixtures/capability_packages/regenerate.py`.
- **Stripped from goldens:** volatile producer fields (`training_version`, `producer`, path sidecars) so version bumps do not force golden churn.
- **Manual edits:** discouraged; change builders, then regenerate.

## Regression layers

1. Golden key/shape tests (this tree)
2. Parametrized builder tests for all `TRAINING_IDS`
3. Publish layout tests
4. Optional live sibling ingest (`aiodoo-validation` / `aiodoo-model`) when present
