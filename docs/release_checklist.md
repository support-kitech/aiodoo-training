# Release Checklist

Checklist for every `aiodoo-training` version bump (including patches).

## Pre-release

- [ ] Change fits [MAINTENANCE.md](MAINTENANCE.md) (PATCH / MINOR / MAJOR)
- [ ] ADR written if required
- [ ] No sibling-repo imports in production code
- [ ] No product composition / registry / certification logic added
- [ ] Capability Package / ArtifactBundle contracts unchanged unless intentionally versioned

## Version & changelog

- [ ] `aiodoo_training/__init__.py` `__version__` updated
- [ ] `CHANGELOG.md` has a top section `## [X.Y.Z] — YYYY-MM-DD`
- [ ] README status / freeze pointers still accurate
- [ ] `docs/repository_freeze.md` still describes the freeze line correctly

## Documentation

- [ ] Lifecycle / ownership / terminology still match shipped behavior
- [ ] No docs claiming unimplemented features as shipped
- [ ] Fixture regenerator still matches builders if metadata changed
  (`python3 tests/fixtures/capability_packages/regenerate.py`)

## Quality gates

```bash
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m mypy aiodoo_training
python3 -m coverage run -m pytest
python3 -m coverage report -m --fail-under=80
```

- [ ] ruff clean
- [ ] mypy clean (infrastructure ignored per policy)
- [ ] all tests passing
- [ ] coverage ≥80%
- [ ] contract tests green (`tests/contract/`)

## Distribution note

This repository is **not** published to PyPI. Current SoT is branch **`main`** plus CHANGELOG. Git tags were removed ecosystem-wide; do not require a tag to consume.
Run from source:

```bash
python3 train.py --config configs/training/<capability>/experiment.yaml
```

## Post-release

- [ ] Tag references CHANGELOG section
- [ ] Sibling docs updated only if they consume changed contracts (separate PRs)

## See also

- [repository_freeze.md](repository_freeze.md)
- [MAINTENANCE.md](MAINTENANCE.md)
- [SMOKE.md](SMOKE.md)
