# FP2-native fixture corpora (TR-3 / TR-4)

Tracked golden fixtures for Canonical Training Contract v1.0.0.

Regenerate (deterministic):

```bash
PYTHONPATH=. python3 -m aiodoo_training.system_training_contract.generators.cli \
  --output-dir fixtures/fp2
```

Quality evaluation (TR-4):

```bash
PYTHONPATH=. python3 -m aiodoo_training.system_training_contract.quality.cli \
  --corpus fixtures/fp2
```

Also emitted to `aiodoo-datasets/datasets/fp2/` (separated from legacy production JSONL).

`quality_negatives.jsonl` is **quality-only** — never train on it.

Do not train adapters from these alone without TR-5+ authorization and coverage expansion.
