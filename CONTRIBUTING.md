# Contributing

Internal AIODOO repository — run from source; do **not** `pip install -e .`.

## Setup

```bash
python3 -m pip install -r requirements/dev.txt
# optional GPU/HF training extras:
python3 -m pip install -r requirements/train.txt
```

## Quality gates

```bash
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m mypy aiodoo_training
python3 -m coverage run -m pytest
python3 -m coverage report -m --fail-under=80
```

## Architecture rules

- Phases 0–7 and Capability Package protocol are **frozen** — see
  [docs/repository_freeze.md](docs/repository_freeze.md).
- Prefer bugfixes and additive extensions behind registries/ports.
- Never import `aiodoo_model` or `aiodoo_validation` from production code.
- Start with [docs/terminology.md](docs/terminology.md) and
  [docs/lifecycle.md](docs/lifecycle.md).

## Capability Package goldens

```bash
python3 tests/fixtures/capability_packages/regenerate.py
```

Policy: [tests/fixtures/capability_packages/README.md](tests/fixtures/capability_packages/README.md).

## Releases

Follow [docs/release_checklist.md](docs/release_checklist.md).
Maintenance policy: [docs/MAINTENANCE.md](docs/MAINTENANCE.md).
