# aiodoo-training — Audit Resolution (v2.0.0)

| Audit Finding | Category | Decision | Action | Reason | Implementation Required? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| Still `__version__=1.0.1` while siblings tagged v2.0.0 | **Production Blocker** | Fix | Bump to 2.0.0; CHANGELOG; docs; tag HEAD | Ecosystem release alignment | **YES** |
| README lists evaluate.py/export.py as if fully working | **Documentation** | Fix | Mark deferred CLI wrappers honestly | Docs vs NotImplemented | **YES** |
| `.gitignore` `/adapters/` fixture fix already on HEAD | **Bug** | Verify | Keep rooted ignore; fixtures tracked | Already fixed post-v1.0.1 | **NO** (done) |
| `cmd_merge` / merge NotImplementedError | **Future Work** | Leave | Document only | Roadmap | **NO** |
| HFExporter always stub | **Intentional** | Leave | Document honestly | Layout export vs real PEFT | **NO** |
| No context validation profile | **Out of Scope** | Leave | Owned by validation | Boundary | **NO** |
| Sparse approval/conversation/evaluation data | **Out of Scope** | Leave | Owned by datasets | Boundary | **NO** |
| Quality gates green | **Intentional** | Keep | No change | Already production-ready in-boundary | **NO** |
