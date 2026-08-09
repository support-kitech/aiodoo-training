# AT-1 — Adapter Training Pipeline Audit

**Living.** Audit of training machinery readiness after TR-7 data readiness.  
**Does not train.** Does not download foundation models. Does not modify
`controlled_batch_2`.

---

## Verdict

**READY_WITH_REQUIRED_FIXES**

| Layer | Status |
|-------|--------|
| Data (`controlled_batch_2`) | **READY_FOR_TRAINING** (TR-7) |
| Training engine (pipeline / HF trainer / LoRA / planner) | **CONFIGURATION + stub EXECUTION ready** |
| FP2 pack → train loader | **NOT wired** (P1) |
| HF PEFT weight export | **Stub layout only** (P1) |
| Controlled smoke / production FP2 train | **Blocked until AT-2 fixes** |

---

## Corpus verification (source)

`aiodoo-datasets/datasets/fp2/controlled_batch_2/`:

| Field | Value |
|-------|------:|
| Version | `fp2-controlled-2.0.0-tr7` |
| Native records | 1386 |
| Contract | `1.0.0` |
| Checksum | match |
| Development pack | 1004 |
| Reasoning pack | 1078 |
| state / DC / loop | 77 / 77 / 77 |
| Negatives in packs/splits | none |

---

## Architecture (source)

```
train.py / cmd_train
  → run_train_from_config
  → Phase4 pipeline
     ValidateConfig → BootstrapDeterminism → ResolveExecution
     → AssembleDatasets → Tokenize → LoadModel → ApplyAdaptation
     → PlanPacking → PlanCurriculum → CreateTrainer → RestoreCheckpoint
     → Train → Evaluate → Export → Finalize
```

- **Entry:** `train.py`, `aiodoo_training.cli.commands.cmd_train`,
  `application/train_orchestrator.run_train_from_config`
- **Trainers:** `StubTrainerBackend` (CI), `HFTrainerBackend` (real
  `transformers.Trainer` loop — not a stub)
- **Adapters:** eight **independent** capability configs
  (`coding`/`repair`/`execution`/`context` /
  `planner`/`conversation`/`approval`/`evaluation`).  
  Development / Reasoning **product packages** are owned by `aiodoo-model`
  (docs + `STATUS.md`). Training does **not** emit chained Dev→Reasoning
  adapters.
- **Foundations (FW0):** Development plane
  `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`; Reasoning plane
  `deepseek-ai/deepseek-vl2`. Model choice is config (`model.backend`,
  `model.identifier`).

**Doc vs source:** AT-1 narrative “two adapters” maps to **product planes**;
the train repo implements **eight skill adapters**. Follow source.

---

## Critical blocker — FP2 packs vs dataset loader

FP2 packs are already `TrainingExample` JSONL
(`example_id`, `dataset_type`, `messages`, `metadata`).

Production loading (`JsonlDatasetSource`) always runs Protocol V1
`ExampleFormatter`s (`CodingFormatter` → `aiodoo_contract` projection).

**Evidence:** loading `pack_development.jsonl` with `CodingFormatter` fails:

`coding record is missing 'output'`

Production `configs/training/*/dataset.yaml` still point at legacy
`coding_v1_0.jsonl` etc., not `controlled_batch_2` packs.

---

## Other P1 gaps

1. `HFExporter` (`hf_peft`) always delegates to `StubExporter` stub weights —
   layout/metadata only; not a real PEFT write.
2. `HFCheckpointStore.restore` raises — HF resume needs application-level
   rehydration (stub resume works).
3. Artifact `contract_version` is provider `aiodoo_contract.CONTRACT_VERSION`,
   not System Training Contract `1.0.0` / corpus checksum fields for FP2.

---

## Smoke design (AT-2 — do not run in AT-1)

Reuse existing stub smoke path for CI proof; for FP2-controlled smoke after
fixes:

1. Add identity / pack loader for TrainingExample JSONL (or tiny isolated
   smoke subset under a temp dir — never overwrite batch_2).
2. Wire a smoke experiment config (tiny `max_steps`, stub or small HF model
   already configured — do not invent a new foundation).
3. Implement real PEFT `save_pretrained` in `HFExporter` (or route smoke
   through checkpoint store that already saves PEFT).
4. Prove: load → LoRA → train step → finite loss → checkpoint → adapter
   metadata → resume.

Illustrative command shape (after AT-2 wiring — **not authorized now**):

```bash
export AIODOO_WORKSPACE_ROOT=/path/to/AIODOO
python train.py --config configs/training/<smoke>/experiment.yaml
```

Existing coding smoke docs: `docs/SMOKE.md` (legacy Protocol coding path).

---

## Required before smoke training (AT-2)

| Priority | Fix |
|----------|-----|
| P1 | FP2 TrainingExample pack loader + smoke config pointing at isolated subset of batch_2 packs |
| P1 | Real HF PEFT adapter write (or smoke-proven checkpoint→publish path) |
| P1 | Provenance fields for FP2 contract version + corpus checksum on artifacts |
| P2 | Align pack provider filtering docs with shared-record behavior |
| P2 | HF checkpoint restore rehydration |
| P3 | CLI `cmd_resume` / export polish (`NotImplementedError` today) |

---

## Runtime independence

Runtime repos do not import Training as a dependency (ECO-1). Mentions of
`aiodoo_training` in core are documentation mirrors / boundary tests only.

---

## AT-2 follow-up

AT-2 implemented the three P1 fixes (FP2 loader, real PEFT export, provenance).
Smoke attempt: **MODEL_UNAVAILABLE** (foundation weights not local).  
See `docs/AT2_FP2_SMOKE.md`.
