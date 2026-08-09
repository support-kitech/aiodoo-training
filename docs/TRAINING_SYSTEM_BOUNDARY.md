# Training / System Boundary — Audit and Freeze

**Living.** Source of truth: current repository code.  
**Decision:** Training is **FROZEN FOR NOW**. System development is primary.

Production adapters are intentionally **not** generated. Path smokes (AT-2…AT-7.8)
prove the Training pipeline can produce PEFT Capability Packages the System can
discover later.

---

## Verdict

| Question | Answer |
|----------|--------|
| Freeze Training? | **Yes — FROZEN FOR NOW** (after minimal handoff alias fix) |
| Boundary correct? | **Yes** — filesystem Capability Package; no runtime Training imports |
| Future adapter without redesign? | **Yes** — place under `adapters_root` → discover → validate → optional attach |
| Foundation-only preserved? | **Yes** — empty adapters root is success; `adapters_required=false` |
| Minimal fix applied | System reads `foundation_model_id`; Training also emits `foundation_hub_id` |

---

## Canonical handoff

```text
Training (aiodoo-training)
  → PEFT export + Drive publish (artifact.json + adapter weights)
  → optional aiodoo-validation certify + aiodoo-model registry
  → place under {model_root}/adapters/ (or AIODOO_ADAPTERS_ROOT)
System (aiodoo-core)
  → AdapterDiscovery.scan
  → AdapterCompatibility.validate
  → AdapterRegistry / AdapterRuntime.resolve
  → ProviderSystem / DualRuntime try_attach_adapter
  → else foundation-only
```

Authoritative package layout:

```text
{adapters_root}/aiodoo-<capability>/
  adapter_config.json
  adapter_model.safetensors|bin
  artifact.json
  manifest.json          # optional diagnostics
```

System must **never** import `aiodoo_training` (ECO-1 tests enforce).

---

## Known deferred (System-side, not Training unblockers)

1. Contract default Hub IDs still DeepSeek; live install uses Qwen — Foundation
   Profile paths select foundations, but adapter hub validation still compares to
   contract Hub constants until System aligns validation with profile hubs.
2. Certification is owned by `aiodoo-validation`; Training does not run it.
3. `context` skill lacks `CapabilityName` in some contract surfaces.
4. VS Code DX and product vertical-slice polish are System work.

---

## Later Training (when authorized)

```text
Dataset readiness → full train → validation → certification
  → packaging → registry → place under adapters_root → System adoption
```

Do not start automatically.
