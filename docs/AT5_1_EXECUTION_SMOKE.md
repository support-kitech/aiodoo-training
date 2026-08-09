# AT-5.1 — Execution Training Path Smoke

**Living.** Corrects AT-5 resource scale (stopped 64-step CPU attempt).  
Proves Execution provider path with a **4-step** smoke.  
**Does not** train the full 522-record Execution population.  
**Does not** certify. **Does not** merge with Coding/Repair.

---

## Verdict

**AT5_1_TRAINING_PATH_PASS**

| Layer | Value |
|-------|-------|
| DATA population | eligible **640**; train **522** / val **63** / test **55** |
| SMOKE population | train **16** / val **4** (derived subset only) |
| TRAINING | **4** real steps; `train_loss=1.268` (finite) |

---

## Why full AT-5 was stopped

Provider selection was correct. `max_steps: 64` on 3B fp32 CPU over the full
522-train load was **resource-scale**, not a data defect. AT-5.1 replaces that
with a path smoke. Authoritative `artifacts/at5_execution/data/*` remains
untouched (522 train). Incomplete AT-5 `checkpoint-32` (if present) is not resumed.

---

## Smoke selection

Deterministic quotas by `record_type` then capability round-robin
(`sha256(record_id)`). Strict subset of Execution train/val splits.

See `artifacts/at5_execution/smoke/smoke_manifest.json` for record IDs.

Config: `configs/training/at5_execution/experiment_at51.yaml`  
Outputs: `artifacts/at5_execution/smoke/{checkpoints,export}/`

---

## Not claimed

- Full Execution adapter trained  
- Production ready / certified  
- Merge with Coding or Repair  

---

## Next

STOP. Await authorization before full Execution training or AT-6 (context).
